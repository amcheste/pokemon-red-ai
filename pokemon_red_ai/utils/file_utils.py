"""
File and directory utilities for Pokemon Red RL.

This module provides utilities for file operations, directory management,
ROM handling, and cleanup operations.
"""

import os
import shutil
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import pickle

logger = logging.getLogger(__name__)


def create_directories(base_path: Union[str, Path]) -> Dict[str, Path]:
    """
    Create standard directory structure for Pokemon Red RL training.

    Args:
        base_path: Base directory path

    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_path = Path(base_path)

    directories = {
        'base': base_path,
        'models': base_path / 'models',
        'logs': base_path / 'logs',
        'tensorboard': base_path / 'tensorboard',
        'monitor': base_path / 'monitor',
        'configs': base_path / 'configs',
        'backups': base_path / 'backups',
        'evaluations': base_path / 'evaluations',
        'screenshots': base_path / 'screenshots',
        'data': base_path / 'data'
    }

    created_dirs = []
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(name)
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise

    logger.info(f"Created directory structure in {base_path}")
    logger.debug(f"Created directories: {', '.join(created_dirs)}")

    return directories


def cleanup_rom_save_files(rom_path: Union[str, Path]) -> List[str]:
    """
    Remove ROM save files to ensure clean training starts.

    Args:
        rom_path: Path to ROM file

    Returns:
        List of removed files
    """
    rom_path = Path(rom_path)
    removed_files = []

    # Common save file extensions used by PyBoy
    save_extensions = ['.ram', '.sav', '.rtc', '.state']

    for ext in save_extensions:
        save_file = rom_path.with_suffix(rom_path.suffix + ext)
        if save_file.exists():
            try:
                save_file.unlink()
                removed_files.append(str(save_file))
                logger.debug(f"Removed save file: {save_file}")
            except Exception as e:
                logger.warning(f"Could not remove {save_file}: {e}")

    # Also check for files with just the extension (no original extension)
    for ext in save_extensions:
        save_file = rom_path.with_suffix(ext)
        if save_file.exists():
            try:
                save_file.unlink()
                removed_files.append(str(save_file))
                logger.debug(f"Removed save file: {save_file}")
            except Exception as e:
                logger.warning(f"Could not remove {save_file}: {e}")

    if removed_files:
        logger.info(f"Cleaned up {len(removed_files)} ROM save files")
    else:
        logger.debug("No ROM save files to clean up")

    return removed_files


def backup_model(model_path: Union[str, Path], backup_dir: Union[str, Path]) -> Path:
    """
    Create a backup of a trained model with timestamp.

    Args:
        model_path: Path to model file to backup
        backup_dir: Directory to store backups

    Returns:
        Path to backup file
    """
    model_path = Path(model_path)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{model_path.stem}_{timestamp}{model_path.suffix}"
    backup_path = backup_dir / backup_name

    try:
        shutil.copy2(model_path, backup_path)
        logger.info(f"Model backed up: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup model: {e}")
        raise


def cleanup_old_backups(backup_dir: Union[str, Path],
                       max_backups: int = 10,
                       pattern: str = "*.zip") -> List[Path]:
    """
    Remove old backup files, keeping only the most recent ones.

    Args:
        backup_dir: Directory containing backups
        max_backups: Maximum number of backups to keep
        pattern: File pattern to match for cleanup

    Returns:
        List of removed backup files
    """
    backup_dir = Path(backup_dir)

    if not backup_dir.exists():
        logger.debug(f"Backup directory does not exist: {backup_dir}")
        return []

    # Find all backup files matching pattern
    backup_files = list(backup_dir.glob(pattern))

    if len(backup_files) <= max_backups:
        logger.debug(f"Only {len(backup_files)} backups found, no cleanup needed")
        return []

    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Remove oldest files
    files_to_remove = backup_files[max_backups:]
    removed_files = []

    for file_path in files_to_remove:
        try:
            file_path.unlink()
            removed_files.append(file_path)
            logger.debug(f"Removed old backup: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove backup {file_path}: {e}")

    if removed_files:
        logger.info(f"Cleaned up {len(removed_files)} old backup files")

    return removed_files


def save_training_metadata(save_path: Union[str, Path],
                          metadata: Dict[str, Any]) -> Path:
    """
    Save training metadata to JSON file.

    Args:
        save_path: Path to save metadata file
        metadata: Metadata dictionary to save

    Returns:
        Path to saved metadata file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp and version info
    enhanced_metadata = metadata.copy()
    enhanced_metadata.update({
        'saved_at': datetime.now().isoformat(),
        'pokemon_red_ai_version': '0.1.0',  # You can import this from __init__.py
        'metadata_version': '1.0'
    })

    try:
        with open(save_path, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)

        logger.info(f"Training metadata saved: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to save training metadata: {e}")
        raise


def load_training_metadata(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load training metadata from JSON file.

    Args:
        metadata_path: Path to metadata file

    Returns:
        Loaded metadata dictionary
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Training metadata loaded: {metadata_path}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to load training metadata: {e}")
        return {}


def get_disk_usage(path: Union[str, Path]) -> Dict[str, float]:
    """
    Get disk usage statistics for a directory.

    Args:
        path: Directory path to analyze

    Returns:
        Dictionary with disk usage info (in MB)
    """
    path = Path(path)

    if not path.exists():
        return {'total_size': 0, 'file_count': 0}

    total_size = 0
    file_count = 0

    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)

        return {
            'total_size_mb': round(total_size_mb, 2),
            'total_size_bytes': total_size,
            'file_count': file_count,
            'avg_file_size_mb': round(total_size_mb / max(file_count, 1), 2)
        }

    except Exception as e:
        logger.error(f"Error calculating disk usage for {path}: {e}")
        return {'error': str(e)}


def find_rom_files(search_path: Union[str, Path],
                  extensions: List[str] = None) -> List[Path]:
    """
    Find ROM files in a directory.

    Args:
        search_path: Directory to search
        extensions: List of ROM file extensions to look for

    Returns:
        List of found ROM file paths
    """
    if extensions is None:
        extensions = ['.gb', '.gbc', '.gba', '.rom']

    search_path = Path(search_path)
    rom_files = []

    if not search_path.exists():
        logger.warning(f"Search path does not exist: {search_path}")
        return rom_files

    for ext in extensions:
        pattern = f"*{ext}"
        found_files = list(search_path.glob(pattern))
        rom_files.extend(found_files)

    # Also search recursively in subdirectories
    for ext in extensions:
        pattern = f"**/*{ext}"
        found_files = list(search_path.glob(pattern))
        rom_files.extend(found_files)

    # Remove duplicates and sort
    rom_files = sorted(list(set(rom_files)))

    logger.info(f"Found {len(rom_files)} ROM files in {search_path}")
    for rom_file in rom_files:
        logger.debug(f"  Found ROM: {rom_file}")

    return rom_files


def validate_rom_file(rom_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate ROM file and return information about it.

    Args:
        rom_path: Path to ROM file

    Returns:
        Dictionary with ROM validation info
    """
    rom_path = Path(rom_path)

    validation_info = {
        'exists': rom_path.exists(),
        'readable': False,
        'size_bytes': 0,
        'size_mb': 0.0,
        'extension': rom_path.suffix.lower(),
        'is_pokemon_red': False,
        'valid': False
    }

    if not validation_info['exists']:
        validation_info['error'] = 'File does not exist'
        return validation_info

    try:
        # Check if file is readable
        with open(rom_path, 'rb') as f:
            # Read first few bytes to check if it's a valid file
            header = f.read(16)
            validation_info['readable'] = True

        # Get file size
        file_stat = rom_path.stat()
        validation_info['size_bytes'] = file_stat.st_size
        validation_info['size_mb'] = round(file_stat.st_size / (1024 * 1024), 2)

        # Check if it's likely Pokemon Red (Game Boy ROM should be 1MB or 2MB)
        if validation_info['extension'] in ['.gb', '.gbc']:
            if validation_info['size_bytes'] in [1048576, 2097152]:  # 1MB or 2MB
                validation_info['is_pokemon_red'] = True

        # Basic validation passed
        validation_info['valid'] = (
            validation_info['readable'] and
            validation_info['size_bytes'] > 0 and
            validation_info['extension'] in ['.gb', '.gbc', '.gba', '.rom']
        )

        logger.debug(f"ROM validation: {rom_path} - Valid: {validation_info['valid']}")

    except Exception as e:
        validation_info['error'] = str(e)
        logger.error(f"ROM validation failed for {rom_path}: {e}")

    return validation_info


def compress_directory(source_dir: Union[str, Path],
                      output_path: Union[str, Path],
                      compression_type: str = 'zip') -> Path:
    """
    Compress a directory to an archive file.

    Args:
        source_dir: Directory to compress
        output_path: Path for output archive
        compression_type: Type of compression ('zip', 'tar', 'tar.gz')

    Returns:
        Path to created archive
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if compression_type == 'zip':
            shutil.make_archive(
                str(output_path.with_suffix('')),
                'zip',
                str(source_dir)
            )
            final_path = output_path.with_suffix('.zip')

        elif compression_type == 'tar':
            shutil.make_archive(
                str(output_path.with_suffix('')),
                'tar',
                str(source_dir)
            )
            final_path = output_path.with_suffix('.tar')

        elif compression_type == 'tar.gz':
            shutil.make_archive(
                str(output_path.with_suffix('')),
                'gztar',
                str(source_dir)
            )
            final_path = output_path.with_suffix('.tar.gz')

        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")

        logger.info(f"Directory compressed: {source_dir} -> {final_path}")
        return final_path

    except Exception as e:
        logger.error(f"Failed to compress directory: {e}")
        raise


def extract_archive(archive_path: Union[str, Path],
                   extract_to: Union[str, Path]) -> Path:
    """
    Extract an archive file to a directory.

    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to

    Returns:
        Path to extraction directory
    """
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive file not found: {archive_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        shutil.unpack_archive(str(archive_path), str(extract_to))
        logger.info(f"Archive extracted: {archive_path} -> {extract_to}")
        return extract_to

    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        raise


def safe_pickle_save(obj: Any, file_path: Union[str, Path]) -> Path:
    """
    Safely save object to pickle file with backup.

    Args:
        obj: Object to pickle
        file_path: Path to save pickle file

    Returns:
        Path to saved pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary file first
    temp_path = file_path.with_suffix('.tmp')

    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        # If original file exists, create backup
        if file_path.exists():
            backup_path = file_path.with_suffix('.bak')
            shutil.move(str(file_path), str(backup_path))

        # Move temporary file to final location
        shutil.move(str(temp_path), str(file_path))

        logger.debug(f"Object pickled safely: {file_path}")
        return file_path

    except Exception as e:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to pickle object: {e}")
        raise


def safe_pickle_load(file_path: Union[str, Path]) -> Any:
    """
    Safely load object from pickle file with fallback to backup.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object
    """
    file_path = Path(file_path)

    # Try to load main file
    if file_path.exists():
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            logger.debug(f"Object loaded from pickle: {file_path}")
            return obj
        except Exception as e:
            logger.warning(f"Failed to load main pickle file: {e}")

    # Try backup file
    backup_path = file_path.with_suffix('.bak')
    if backup_path.exists():
        try:
            with open(backup_path, 'rb') as f:
                obj = pickle.load(f)
            logger.warning(f"Object loaded from backup pickle: {backup_path}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load backup pickle file: {e}")

    raise FileNotFoundError(f"No valid pickle file found: {file_path}")


def monitor_directory_size(directory: Union[str, Path],
                          max_size_mb: float = 1000.0) -> Dict[str, Any]:
    """
    Monitor directory size and warn if it exceeds threshold.

    Args:
        directory: Directory to monitor
        max_size_mb: Maximum size threshold in MB

    Returns:
        Dictionary with monitoring info
    """
    directory = Path(directory)

    if not directory.exists():
        return {'error': 'Directory does not exist', 'size_mb': 0}

    usage_info = get_disk_usage(directory)
    current_size = usage_info.get('total_size_mb', 0)

    monitor_info = {
        'directory': str(directory),
        'current_size_mb': current_size,
        'max_size_mb': max_size_mb,
        'usage_percent': (current_size / max_size_mb) * 100,
        'exceeds_threshold': current_size > max_size_mb,
        'file_count': usage_info.get('file_count', 0)
    }

    if monitor_info['exceeds_threshold']:
        logger.warning(f"Directory size exceeds threshold: {directory}")
        logger.warning(f"  Current: {current_size:.1f}MB, Max: {max_size_mb:.1f}MB")
    elif monitor_info['usage_percent'] > 80:
        logger.warning(f"Directory size approaching threshold: {directory}")
        logger.warning(f"  Current: {current_size:.1f}MB ({monitor_info['usage_percent']:.1f}%)")

    return monitor_info


def cleanup_temp_files(base_dir: Union[str, Path],
                      patterns: List[str] = None) -> List[Path]:
    """
    Clean up temporary files in a directory.

    Args:
        base_dir: Base directory to clean
        patterns: List of file patterns to clean (default: common temp patterns)

    Returns:
        List of removed files
    """
    if patterns is None:
        patterns = ['*.tmp', '*.temp', '*.log.*', '*.bak', '*.old', '*~']

    base_dir = Path(base_dir)
    removed_files = []

    if not base_dir.exists():
        return removed_files

    for pattern in patterns:
        for file_path in base_dir.rglob(pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    removed_files.append(file_path)
                    logger.debug(f"Removed temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {file_path}: {e}")

    if removed_files:
        logger.info(f"Cleaned up {len(removed_files)} temporary files")

    return removed_files


def get_project_info(project_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a Pokemon Red RL project directory.

    Args:
        project_dir: Project directory path

    Returns:
        Dictionary with project information
    """
    project_dir = Path(project_dir)

    info = {
        'project_dir': str(project_dir),
        'exists': project_dir.exists(),
        'created_at': None,
        'last_modified': None,
        'total_size_mb': 0,
        'directories': {},
        'models': [],
        'configs': [],
        'logs': [],
        'rom_files': []
    }

    if not project_dir.exists():
        return info

    try:
        # Basic directory info
        stat_info = project_dir.stat()
        info['created_at'] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        info['last_modified'] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()

        # Directory analysis
        usage = get_disk_usage(project_dir)
        info['total_size_mb'] = usage.get('total_size_mb', 0)
        info['file_count'] = usage.get('file_count', 0)

        # Check standard subdirectories
        standard_dirs = ['models', 'logs', 'tensorboard', 'monitor', 'configs']
        for dir_name in standard_dirs:
            dir_path = project_dir / dir_name
            if dir_path.exists():
                dir_usage = get_disk_usage(dir_path)
                info['directories'][dir_name] = {
                    'exists': True,
                    'size_mb': dir_usage.get('total_size_mb', 0),
                    'file_count': dir_usage.get('file_count', 0)
                }
            else:
                info['directories'][dir_name] = {'exists': False}

        # Find models
        model_patterns = ['*.zip', '*.pkl', '*.pt', '*.pth']
        for pattern in model_patterns:
            info['models'].extend(list(project_dir.rglob(pattern)))

        # Find configs
        config_patterns = ['*.yaml', '*.yml', '*.json']
        for pattern in config_patterns:
            info['configs'].extend(list(project_dir.rglob(pattern)))

        # Find log files
        log_patterns = ['*.log', '*.txt']
        for pattern in log_patterns:
            found_logs = list((project_dir / 'logs').glob(pattern)) if (project_dir / 'logs').exists() else []
            info['logs'].extend(found_logs)

        # Find ROM files
        info['rom_files'] = find_rom_files(project_dir)

        # Convert Path objects to strings for JSON serialization
        info['models'] = [str(p) for p in info['models']]
        info['configs'] = [str(p) for p in info['configs']]
        info['logs'] = [str(p) for p in info['logs']]
        info['rom_files'] = [str(p) for p in info['rom_files']]

    except Exception as e:
        info['error'] = str(e)
        logger.error(f"Error analyzing project directory {project_dir}: {e}")

    return info