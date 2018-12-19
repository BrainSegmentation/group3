from imgaug import augmenters as iaa
"""this file contains the definition of the augmenter function used in maskrcnn training."""


def aug():
	return iaa.OneOf(
		[iaa.Affine(rotate=(-90,90)),                               # affine random rotation
        iaa.Affine(scale=(.5,2)),                                   # affine random rescaling
        iaa.Affine(translate_px={"x": (-50, 100), "y": (-50, 50)}), # affine random translation
		iaa.Fliplr(0.5),                                            # flipping
        iaa.Noop()]
        )