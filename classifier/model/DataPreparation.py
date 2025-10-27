import os.path
import tensorflow as tf
from utils.preprocessing import prepare_dataset
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    This class makes semi-automatic the creation of datasets.
    """
    def __init__(self,
                 root_folder,
                 img_size,
                 color_mode,
                 batch_size,
                 class_mode: str = 'binary',
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 validation_split: Optional[float] = None,
                 subset: Optional[str] = None,
                 cache: bool = True,
                 prefetch: bool = True,
                 recursive: bool = False):

        self.root_folder = root_folder
        self.img_size = img_size
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.class_mode = class_mode
        # nuevos parámetros opcionales
        self.shuffle = shuffle
        self.seed = seed
        self.validation_split = validation_split
        self.subset = subset
        self.cache = cache
        self.prefetch = prefetch
        self.recursive = recursive

    def create_datasets_from_subdirectories(self) -> Dict[str, tf.data.Dataset]:
        # Validar carpeta raíz
        root = Path(self.root_folder)
        if not root.exists():
            logger.error("Root folder does not exist: %s", self.root_folder)
            raise FileNotFoundError(f"Root folder does not exist: {self.root_folder}")

        # Listar sólo subdirectorios
        subdirectories = [p for p in root.iterdir() if p.is_dir()]
        datasets = {}
        if not subdirectories:
            # Intentar cargar el root_folder como un dataset único
            try:
                ds = self.create_dataset(str(root))
                datasets[root.name] = ds
                logger.info("Loaded single dataset from root folder.")
            except Exception as e:
                logger.exception("Failed to load dataset from root folder: %s", e)
            return datasets

        for sub in subdirectories:
            path = str(sub)
            try:
                ds = self.create_dataset(path)
                datasets[sub.name] = ds
            except Exception as e:
                logger.exception("Skipping dataset at %s due to error: %s", path, e)
                # continuar con los demás datasets

        logger.info('Data normalization done!')
        return datasets

    def create_dataset(self, path: str) -> tf.data.Dataset:
        # Detectar número de clases (subcarpetas con archivos) para ajustar label_mode
        p = Path(path)
        class_dirs = [d for d in p.iterdir() if d.is_dir() and any(f for f in d.rglob('*') if f.is_file())]
        classes_count = len(class_dirs)

        # Decidir label_mode adaptivamente
        label_mode = self.class_mode
        if classes_count == 0:
            # no hay subclases: sin etiquetas
            label_mode = None
            logger.info("No class subdirectories found in %s — using label_mode=None", path)
        else:
            if self.class_mode == 'binary' and classes_count != 2:
                label_mode = 'categorical'
                logger.info("class_mode='binary' but %d classes found in %s — switching to 'categorical'", classes_count, path)

        try:
            dataset = tf.keras.utils.image_dataset_from_directory(
                directory=path,
                image_size=self.img_size,
                color_mode=self.color_mode,
                batch_size=self.batch_size,
                label_mode=label_mode,
                shuffle=self.shuffle,
                seed=self.seed,
                validation_split=self.validation_split,
                subset=self.subset,
                # si recursive es True, incluir subdirectorios profundos
                follow_links=self.recursive
            )
            logger.info('Dataset loaded successfully from %s', path)
            logger.info('Classes found in dataset: %s', getattr(dataset, 'class_names', None))
        except Exception as e:
            logger.exception("Error loading dataset from %s: %s", path, e)
            raise

        # Preparar dataset según si es train o no (mantener compatibilidad con la lógica previa)
        try:
            is_training = 'train' in path.lower()
            dataset = prepare_dataset(dataset, training=is_training) if is_training else prepare_dataset(dataset)
        except Exception:
            # Si prepare_dataset falla, no romper todo: loguear y devolver dataset sin procesar
            logger.exception("prepare_dataset failed for %s, returning raw dataset", path)

        # Aplicar cache / prefetch si procede
        try:
            if self.cache:
                dataset = dataset.cache()
            if self.prefetch:
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
        except Exception:
            logger.exception("Error applying cache/prefetch for %s — returning dataset without those optimizations", path)

        return dataset