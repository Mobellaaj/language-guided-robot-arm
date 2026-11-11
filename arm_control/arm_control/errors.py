"""Errors that distinguish temporary perception gaps from invalid tasks."""


class PerceptionNotReady(ValueError):
    """Fresh, unoccluded observations may make this preparation request valid."""
