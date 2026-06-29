"""Extract MNE event arrays from annotated EEG recordings."""

import mne
import numpy as np

def extract_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    """Extract event samples and labels from one annotated raw run.

    Args:
        raw: MNE raw recording with annotations.

    Returns:
        Tuple of ``(events, event_id)`` as returned by MNE.
    """

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id
