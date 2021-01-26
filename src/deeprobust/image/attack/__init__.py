#__init__.py
import logging

from src.deeprobust.image.attack import pgd
from src.deeprobust.image.attack import deepfool
from src.deeprobust.image.attack import fgsm
__all__ = ['pgd', 'fgsm', 'deepfool']

logging.info("import base_attack from attack")
