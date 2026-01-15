import sys
from pathlib import Path

def print_directory_tree(root_path_str, max_files=5):
    """
    Prints the directory tree structure to the console, limiting
    the number of files displayed per directory.
    """
    root_path = Path(root_path_str+"/data")
    
    if not root_path.exists():
        print(f"Error: The path '{root_path}' does not exist.")
        return
    if not root_path.is_dir():
        print(f"Error: The path '{root_path}' is not a directory.")
        return

    print(f"{root_path.name}/")
    _print_tree_recursive(root_path, "", max_files)

def _print_tree_recursive(directory, prefix, max_files):
    """
    Recursive helper function to process directories and files.
    """
    try:
        # Get all items in the directory
        # We sort them to ensure consistent output
        items = list(directory.iterdir())
        items.sort(key=lambda x: x.name.lower())
    except PermissionError:
        print(f"{prefix}└── [Access Denied]")
        return
    except OSError as e:
        print(f"{prefix}└── [Error: {e}]")
        return

    # Separate directories and files
    dirs = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file()]

    # We will display all directories, but limit files
    # The display list contains all dirs + the first 'max_files'
    display_files = files[:max_files]
    hidden_files_count = len(files) - len(display_files)
    
    # Combine them for the loop (Directories first, then files)
    entries_to_show = dirs + display_files
    total_entries = len(entries_to_show)

    for index, entry in enumerate(entries_to_show):
        # Determine if this is the last item visually
        # It is last if it's the last in the list AND there are no hidden files remaining
        is_last_item = (index == total_entries - 1) and (hidden_files_count == 0)
        
        # visual connectors
        connector = "└── " if is_last_item else "├── "
        
        if entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/")
            
            # Prepare the prefix for the next level of recursion
            # If this was the last item, children get empty space, otherwise a vertical bar
            extension = "    " if is_last_item else "│   "
            _print_tree_recursive(entry, prefix + extension, max_files)
        else:
            print(f"{prefix}{connector}{entry.name}")

    # If there are files we didn't show, print a summary line
    if hidden_files_count > 0:
        print(f"{prefix}└── ... ({hidden_files_count} more files)")

if __name__ == "__main__":
    # Default to current directory if no argument provided
    target_dir = "."
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        
    print_directory_tree(target_dir, max_files=5)