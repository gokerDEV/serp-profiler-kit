import os

def normalize_folder(folder: str) -> str:
    """
    Accept both:
      - '3-bot/000'
      - 'scraping/3-bot/000'
    and normalize to '3-bot/000'
    """
    f = str(folder or "").strip().replace("\\", "/")
    if f.startswith("scraping/"):
        f = f[len("scraping/"):]
    return f

def resolve_path(base_dir: str, folder: str, file_name: str, extension: str = ".json") -> str:
    """
    Resolves file path robustly, handling potential .html suffix issues in file_name.
    
    1. Tries {base_dir}/{folder}/{file_name}{extension}
    2. If file_name ends with '.html', tries {base_dir}/{folder}/{file_name_no_html}{extension}
    3. Returns the first existing path, or the default (first option) if neither exists.
    """
    # 1. Standard approach
    path1 = os.path.join(base_dir, folder, f"{file_name}{extension}")
    if os.path.exists(path1):
        return path1
        
    # 2. Try stripping .html (common issue with index file_name vs artifact name)
    if file_name.lower().endswith(".html"):
        clean_name = file_name[:-5] # remove .html
        path2 = os.path.join(base_dir, folder, f"{clean_name}{extension}")
        if os.path.exists(path2):
            return path2
            
    # Default to standard if neither found
    return path1
