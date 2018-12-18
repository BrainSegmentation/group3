from imgaug import augmenters as iaa



def aug():
	return iaa.OneOf(
		[iaa.Affine(rotate=(-90,90)),
        iaa.Affine(scale=(.5,2)),
        iaa.Affine(translate_px={"x": (-50, 100), "y": (-50, 50)}),
		iaa.Fliplr(0.5),
        iaa.Noop()]
        )