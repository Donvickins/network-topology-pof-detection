class InvalidImageException(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__(self.details)

    def __str__(self):
        return self.details

class SiteIdNotFoundInImage(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__(self.details)

    def __str__(self):
        return self.details

class NoSiteId(Exception):
    def __init__(self, details):
        self.details = details
        super().__init__(self.details)

    def __str__(self):
        return self.details