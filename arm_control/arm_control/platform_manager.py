"""Use fresh observed slot occupancy; keep uncertain releases reserved."""
import math
from arm_interfaces.msg import PlatformSlot


class PlatformManager:
    """Track either placement area's observations and reservations independently."""

    def __init__(self, node, layout, surface='platform'):
        self.node = node
        self.layout = layout
        self.surface = surface
        self.slots = layout.placement_slots(surface)
        self.message = None
        self.stability = {}
        self.reserved = {}

    def update(self, message):
        self.message = message
        stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
        for slot in message.slots:
            key = (slot.state, slot.color)
            previous = self.stability.get(slot.name)
            if previous is None or previous[0] != key or stamp <= previous[2]:
                self.stability[slot.name] = (key, stamp, stamp, 1)
            else:
                self.stability[slot.name] = (key, previous[1], stamp, previous[3]+1)

    def fresh_slots(self):
        if self.message is None or self.message.header.frame_id != 'base_link':
            raise ValueError(f'Need {self.surface} observations in base_link. Start the combined launch.')
        if not -0.05 <= self.node.age(self.message.header.stamp) <= 0.5:
            raise ValueError(f'{self.surface.capitalize()} observations are stale. Check camera visibility.')
        slots = {slot.name: slot for slot in self.message.slots}
        if set(slots) != set(self.slots):
            raise ValueError(f'{self.surface.capitalize()} slot layout does not match perception.')
        for name, center in self.slots.items():
            p = slots[name].center
            if max(abs(a-b) for a,b in zip((p.x,p.y,p.z),center)) > 0.003:
                raise ValueError(f'{self.surface.capitalize()} slot coordinates do not match the configured world.')
        return slots

    def stable(self, name):
        item = self.stability.get(name)
        return item is not None and item[3] >= 3 and item[2]-item[1] >= 0.4

    def assert_empty(self, name):
        slot = self.fresh_slots()[name]
        if slot.state != PlatformSlot.EMPTY or not self.stable(name):
            raise ValueError(f'{name} is occupied, hidden, or not stably observed as empty.')

    def available(self):
        slots = self.fresh_slots()
        names = [name for name in self.slots if name not in self.reserved
                 and slots[name].state == PlatformSlot.EMPTY and self.stable(name)]
        if not names:
            raise ValueError(f'No confirmed empty {self.surface} slot. It may be full, hidden, or reserved. '
                             f'Check camera visibility and run inspect {self.surface}.')
        return names

    def reserve(self, name):
        if name in self.reserved:
            raise ValueError(f'{name} is already reserved.')
        self.reserved[name] = 'reserved'

    def release_reservation(self, name):
        # An unverified release is deliberately not forgotten by a cancel/replan.
        if self.reserved.get(name) == 'reserved':
            self.reserved.pop(name)

    def released(self, name):
        self.reserved[name] = 'unverified'

    def verified(self, name, color):
        slot = self.fresh_slots()[name]
        if not (slot.state == PlatformSlot.OCCUPIED and slot.color == color and self.stable(name)):
            return False
        objects = self.node.objects
        if objects is None or objects.header.frame_id != 'base_link' or not -0.05 <= self.node.age(objects.header.stamp) <= 0.5:
            return False
        center = self.slots[name]
        for detection in objects.detections:
            if detection.results and detection.results[0].hypothesis.class_id == color:
                point = detection.results[0].pose.pose.position
                if (math.hypot(point.x-center[0],point.y-center[1]) <= 0.022
                        and abs(point.z-(center[2]+self.layout.cube)) <= 0.005):
                    return True
        return False

    def inspect(self):
        slots = self.fresh_slots()
        for name in list(self.reserved):
            if (self.reserved[name] == 'unverified' and self.stable(name)
                    and slots[name].state in (PlatformSlot.EMPTY, PlatformSlot.OCCUPIED)):
                self.reserved.pop(name)
        labels = {PlatformSlot.UNKNOWN:'unknown', PlatformSlot.EMPTY:'empty', PlatformSlot.OCCUPIED:'occupied'}
        return '; '.join(f'{name}: '+(self.reserved.get(name) or labels.get(slot.state,'unknown'))
                         +(f' ({slot.color})' if slot.color else '') for name,slot in slots.items())
