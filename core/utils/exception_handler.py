class InvalidImageException(Exception):
    def __init__(self, details):
        self.message = "Invalid image" if not details else details
        super().__init__(self.message)

    def __str__(self):
        return self.message

class SiteIdNotFoundInImage(Exception):
    def __init__(self, details):
        self.message = "Site ID not found in image" if not details else details
        super().__init__(self.message)

    def __str__(self):
        return self.message

class NoSiteId(Exception):
    def __init__(self, details):
        self.message = "No site id was provided" if not details else details
        super().__init__(self.message)

    def __str__(self):
        return self.message