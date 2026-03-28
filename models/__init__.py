try:
    from .crnn import EndToEndLPR
except ImportError:
    from crnn import EndToEndLPR

__all__ = ['EndToEndLPR']
