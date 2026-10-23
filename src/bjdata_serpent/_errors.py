class DecodeError(ValueError):
    """The bytes are not a well-formed BJData document, or hold something that cannot be decoded.

    ``offset`` is the byte the problem was found at, and ``code`` serpent's name for it - such as
    ``"unexpected_end"`` - or None when the bytes parse but what they hold does not decode.
    """

    def __init__(self, message, offset=None, code=None):
        super().__init__(message)
        self.offset = offset
        self.code = code


class EncodeError(ValueError):
    """The value cannot be written as BJData: nested too deeply, say, or a table of the wrong shape."""
