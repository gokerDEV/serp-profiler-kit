from tqdm import tqdm

def get_progress_bar(iterable, desc=None, total=None):
    """
    Returns a unified tqdm progress bar.
    
    Args:
        iterable: The iterable to wrap.
        desc: Description of the progress bar.
        total: Total number of items (optional, useful if iterable is valid generator).
    """
    return tqdm(iterable, desc=desc, total=total, ncols=100, dynamic_ncols=True)
