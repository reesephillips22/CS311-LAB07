"""
Lab 7: The Collision Resolver -- completed.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert, or update in place if key already exists."""
        index = hash(key) % len(self._buckets)
        current = self._buckets[index]

        while current is not None:
            if current.key == key:
                current.value = value
                return
            current = current.next

        new_node = _ChainNode(key, value)
        new_node.next = self._buckets[index]
        self._buckets[index] = new_node

        self._count += 1

        if self._count / len(self._buckets) > 0.75:
            old_buckets = self._buckets
            self._buckets = [None] * (len(old_buckets) * 2)
            self._count = 0

            for node in old_buckets:
                current = node

                while current is not None:
                    self.insert(current.key, current.value)
                    current = current.next

    def get(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)
        current = self._buckets[index]

        while current is not None:
            if current.key == key:
                return current.value

            current = current.next

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove key. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)
        current = self._buckets[index]
        previous = None

        while current is not None:
            if current.key == key:
                if previous is None:
                    self._buckets[index] = current.next
                else:
                    previous.next = current.next

                self._count -= 1
                return

            previous = current
            current = current.next

        raise KeyError(key)


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize once the load factor would exceed 0.7."""

        if (self._count + 1) / len(self._keys) > 0.7:
            old_keys = self._keys
            old_values = self._values

            self._keys = [None] * (len(old_keys) * 2)
            self._values = [None] * (len(old_values) * 2)
            self._count = 0

            for i in range(len(old_keys)):
                if (
                    old_keys[i] is not None
                    and old_keys[i] is not _TOMBSTONE
                ):
                    self.insert(old_keys[i], old_values[i])

        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i) % len(self._keys)

            if self._keys[probe_index] is None:
                self._keys[probe_index] = key
                self._values[probe_index] = value
                self._count += 1
                return

            if self._keys[probe_index] is _TOMBSTONE:
                self._keys[probe_index] = key
                self._values[probe_index] = value
                self._count += 1
                return

            if self._keys[probe_index] == key:
                self._values[probe_index] = value
                return

            i += 1

    def search(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i) % len(self._keys)

            if self._keys[probe_index] is None:
                raise KeyError(key)

            if self._keys[probe_index] == key:
                return self._values[probe_index]  # type: ignore

            i += 1

    def delete(self, key: K) -> None:
        """Remove key using a tombstone."""
        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i) % len(self._keys)

            if self._keys[probe_index] is None:
                raise KeyError(key)

            if self._keys[probe_index] == key:
                self._keys[probe_index] = _TOMBSTONE
                self._values[probe_index] = None
                self._count -= 1
                return

            i += 1


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @staticmethod
    def _is_prime(number: int) -> bool:
        """Return True if number is prime."""
        if number < 2:
            return False

        divisor = 2

        while divisor * divisor <= number:
            if number % divisor == 0:
                return False
            divisor += 1

        return True

    @classmethod
    def _next_prime(cls, number: int) -> int:
        """Return the next prime number at or above number."""
        while not cls._is_prime(number):
            number += 1

        return number

    def _resize(self) -> None:
        """Grow the table and rehash all existing entries."""
        old_keys = self._keys
        old_values = self._values

        new_size = self._next_prime(len(old_keys) * 2)

        self._keys = [None] * new_size
        self._values = [None] * new_size
        self._count = 0

        for i in range(len(old_keys)):
            if (
                old_keys[i] is not None
                and old_keys[i] is not _TOMBSTONE
            ):
                self.insert(old_keys[i], old_values[i])

    def insert(self, key: K, value: V) -> None:
        """Resize once the load factor would exceed 0.7."""

        if (self._count + 1) / len(self._keys) > 0.7:
            self._resize()

        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i * i) % len(self._keys)

            if self._keys[probe_index] is None:
                self._keys[probe_index] = key
                self._values[probe_index] = value
                self._count += 1
                return

            if self._keys[probe_index] is _TOMBSTONE:
                self._keys[probe_index] = key
                self._values[probe_index] = value
                self._count += 1
                return

            if self._keys[probe_index] == key:
                self._values[probe_index] = value
                return

            i += 1

    def search(self, key: K) -> V:
        """Return the value for key. Raise KeyError if missing."""
        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i * i) % len(self._keys)

            if self._keys[probe_index] is None:
                raise KeyError(key)

            if self._keys[probe_index] == key:
                return self._values[probe_index]  # type: ignore

            i += 1

    def delete(self, key: K) -> None:
        """Remove key using a tombstone."""
        index = hash(key) % len(self._keys)
        i = 0

        while True:
            probe_index = (index + i * i) % len(self._keys)

            if self._keys[probe_index] is None:
                raise KeyError(key)

            if self._keys[probe_index] == key:
                self._keys[probe_index] = _TOMBSTONE
                self._values[probe_index] = None
                self._count -= 1
                return

            i += 1