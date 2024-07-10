from django.db import models

class UploadedImage(models.Model):
    original_image = models.ImageField(upload_to='originals/')
    processed_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"Image uploaded on {self.upload_date}"