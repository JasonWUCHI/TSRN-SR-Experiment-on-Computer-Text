import cv2

def blur_img(large_img, idx):
    if idx%3==0:
        blur_img = cv2.GaussianBlur(large_img,(7,7),0)
    elif idx%3==1:
        blur_img = cv2.blur(large_img,(4,4),0)
    elif idx%3==2:
        small_to_large_image_size_ratio = 0.4
        small_img = cv2.resize(large_img, # original image
                            (0,0), # set fx and fy, not the final size
                            fx=small_to_large_image_size_ratio, 
                            fy=small_to_large_image_size_ratio, )
        blur_img = cv2.resize(small_img, # original image
                            (256,256) # set fx and fy, not the final size
                            )
    cv2.imwrite('blur_imgs/%d.png'%idx , blur_img)

if __name__ == '__main__':
    for img_name in range(1,1001):
        large_img = cv2.imread('imgs/%d.png'%img_name)
        blur_img(large_img, img_name)