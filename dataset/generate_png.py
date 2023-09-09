from PIL import Image, ImageDraw, ImageFont
import os, random
import numpy as np
import cv2
import re

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def blur_img(large_img, blur_rate, idx):
    small_to_large_image_size_ratio = blur_rate
    small_img = cv2.resize(large_img, # original image
                        (0,0), # set fx and fy, not the final size
                        fx=small_to_large_image_size_ratio, 
                        fy=small_to_large_image_size_ratio, )
    blur_img = cv2.resize(small_img, # original image
                        (256,256) # set fx and fy, not the final size
                        )
    cv2.imwrite('blur_imgs/%d.png'%idx , blur_img)

def get_png(text, img_name):
    width = 256
    height = 256
    font_size, gap_x, gap_y = [15,20,28], [9,13,19], [21, 30, 39]
    char_length_limit, line_limit = [37,25,21], [11,8,6]
    font_seed = random.randint(0,2)
    font = ImageFont.truetype("arial.ttf", size=font_size[font_seed])

    img = Image.new('RGB', (width, height), color='white')
    imgDraw = ImageDraw.Draw(img)

    y = 10
    color = [(0,0,0), (36,25,171), (171, 49, 36)]
    text_list = text.split()
    text_list = [text for text in text_list if not has_numbers(text)]

    for _ in range(line_limit[font_seed]):
        start = random.randint(0, len(text_list)-10)
        check = 0
        num = 0
        while check<char_length_limit[font_seed]:
            if check+len(text_list[start+num])<char_length_limit[font_seed]:
                check += len(text_list[start+num])+1
                num += 1
            else:
                break
        word = text_list[start:start+num]
    
        x = 10
        imgDraw.text((x, y), ' '.join(word), font = font, fill=(0,0,0))

        y = y+gap_y[font_seed]
        img.save("imgs/" + str(img_name) + ".png")
        
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        blur_rate = 0.4 if font_seed==0 else 0.3
        blur_img(open_cv_image,blur_rate, img_name )


if __name__ == '__main__':
    text_path = "/Users/jason543wu/Desktop/WikiResearch/shared/pair_data_organized_new/"
    img_name = 1
    for fname in os.listdir(text_path)[:100]:
        with open(text_path + fname) as f:
            lines = f.readlines()
            if len(lines[0])<1000:
                continue
            for i in range(12):
                get_png(lines[0], img_name)
                img_name += 1
                if img_name>1000:
                    sys.exit()


'''
word = text[start:start+30].split()
    
        x = 10
        for w in word:
            color_seed = random.randint(1,20)
            if color_seed in [13,14]:
                c = color[1]
            elif color_seed == 15:
                c = color[2]
            else:
                c = color[0]

            imgDraw.text((x, y), w, font = font, fill=c)
            x = x+len(w)*13

        y = y+30
        img.save("imgs/" + str(img_name) + ".png")
'''