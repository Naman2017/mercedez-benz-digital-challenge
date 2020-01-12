import segment_inference
import inference
import cv2


def pred(img):
	seg=segment_inference.segment()
	img=seg.converter(img)
	# cv2.imshow('a',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	infer_=inference.infer()
	flag=(infer_.predict(img))

	### closing 
	infer_.close()
	seg.close()

	return flag