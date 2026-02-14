"""
Service infrastructure for Monarch actors.

Provides:
- ServiceRegistry: Singleton for service discovery
- Service: Manages worker replicas with health tracking and routing
- get_service/register_service: Helper functions for discovery
"""

from monarch.actor import Actor, endpoint, get_or_spawn_controller


class ServiceRegistry(Actor):
    """Singleton registry for service discovery."""

    def __init__(self):
        self.services: dict[str, Actor] = {}
        print("[REGISTRY] ServiceRegistry spawned")

    @endpoint
    def register(self, name: str, service) -> None:
        """Register a service by name."""
        self.services[name] = service
        print(f"[REGISTRY] Registered '{name}'")

    @endpoint
    def get(self, name: str):
        """Get a service by name."""
        if name not in self.services:
            raise KeyError(f"Service '{name}' not found")
        return self.services[name]

    @endpoint
    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self.services.keys())


def _get_registry():
    """Get the singleton ServiceRegistry."""
    return get_or_spawn_controller("service_registry", ServiceRegistry).get()


def get_service(name: str):
    """Get a service by name. Raises KeyError if not found."""
    return _get_registry().get.call_one(name).get()


def register_service(name: str, service) -> None:
    """Register a service so others can find it by name."""
    _get_registry().register.call_one(name, service).get()


def list_services() -> list[str]:
    """List all registered service names."""
    return _get_registry().list_services.call_one().get()


class Service(Actor):
    """Manages a pool of worker replicas with routing and health tracking.

    The Service owns worker lifecycle: it slices a ProcMesh into
    replica-sized chunks and spawns an ActorMesh on each.

    Usage:
        # Spawn Service actor on its own proc
        svc = svc_proc.spawn("my_service", Service,
            service_name="generators",
            worker_class=GeneratorWorker,
            procs=worker_procs,
            procs_per_replica=1,
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
        )

        # Register for discovery
        register_service("generators", svc)

        # Get a replica (round-robin)
        worker, idx = svc.get_replica_with_idx.call_one().get()

        # On failure, mark unhealthy and retry
        svc.mark_unhealthy.call_one(idx).get()
    """

    def __init__(
        self,
        service_name: str,
        worker_class: type,
        procs,
        procs_per_replica: int = 1,
        **worker_kwargs,
    ):
        self.service_name = service_name
        total = len(procs)
        self.num_replicas = total // procs_per_replica

        # Slice ProcMesh and spawn a replica on each chunk
        self.replicas = []
        for i in range(self.num_replicas):
            start = i * procs_per_replica
            end = start + procs_per_replica
            replica_slice = procs.slice(procs=slice(start, end))
            replica = replica_slice.spawn(
                f"replica_{i}", worker_class, **worker_kwargs,
            )
            self.replicas.append(replica)

        self.healthy = set(range(self.num_replicas))
        self.unhealthy: set[int] = set()
        self.next_idx = 0

        print(f"[SERVICE:{service_name}] {self.num_replicas} replicas "
              f"x {procs_per_replica} procs each")

    @endpoint
    def get_replica(self):
        """Get a healthy replica (round-robin selection)."""
        if not self.healthy:
            raise RuntimeError("No healthy replicas available!")
        healthy_list = sorted(self.healthy)
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1
        return self.replicas[healthy_list[idx]]

    @endpoint
    def get_replica_with_idx(self):
        """Get (replica, index). Use the index for mark_unhealthy()."""
        if not self.healthy:
            raise RuntimeError("No healthy replicas available!")
        healthy_list = sorted(self.healthy)
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1
        replica_idx = healthy_list[idx]
        return self.replicas[replica_idx], replica_idx

    @endpoint
    def mark_unhealthy(self, replica_idx: int) -> None:
        """Remove a replica from rotation."""
        if replica_idx in self.healthy:
            self.healthy.discard(replica_idx)
            self.unhealthy.add(replica_idx)
            print(f"[SERVICE:{self.service_name}] Replica {replica_idx} unhealthy. "
                  f"Healthy: {len(self.healthy)}/{self.num_replicas}")

    @endpoint
    def mark_healthy(self, replica_idx: int) -> None:
        """Reinstate a replica."""
        if replica_idx < self.num_replicas:
            self.unhealthy.discard(replica_idx)
            self.healthy.add(replica_idx)

    @endpoint
    def check_health(self) -> dict:
        """Probe unhealthy replicas. Reinstate those that respond to ping()."""
        recovered = []
        still_unhealthy = []
        for replica_idx in list(self.unhealthy):
            try:
                # Use .call() not .call_one() — works for multi-GPU replicas too
                self.replicas[replica_idx].ping.call().get()
                self.unhealthy.discard(replica_idx)
                self.healthy.add(replica_idx)
                recovered.append(replica_idx)
                print(f"[SERVICE:{self.service_name}] Replica {replica_idx} recovered!")
            except Exception:
                still_unhealthy.append(replica_idx)
        return {
            "recovered": recovered,
            "still_unhealthy": still_unhealthy,
            "healthy_count": len(self.healthy),
        }

    @endpoint
    def get_health_status(self) -> dict:
        """Get current health status."""
        return {
            "total": self.num_replicas,
            "healthy": len(self.healthy),
            "unhealthy": len(self.unhealthy),
            "healthy_indices": sorted(self.healthy),
            "unhealthy_indices": sorted(self.unhealthy),
        }

    @endpoint
    def get_all_replicas(self) -> list:
        """Get all replicas (for operations that touch every replica)."""
        return list(self.replicas)

    @endpoint
    def ping(self) -> bool:
        """Health check for the service itself."""
        return True
