from PIL import Image, ImageDraw, ImageFont
import os, random
import numpy as np
import cv2
import re
import pickle

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
    cv2.imwrite('small_imgs/%d.png'%idx, small_img)

def get_png(text, img_name, words_dict):
    width = 256
    height = 256
    font_seed = random.choice(range(0,4))
    font_size, scale_size = [16,18,20,22,24], [0.4,0.4,0.3,0.3,0.3]
    font = ImageFont.truetype("arial.ttf", size=font_size[font_seed])
    img = Image.new('RGB', (width, height), color='white')
    imgDraw = ImageDraw.Draw(img)

    color = [(0,0,0), (36,25,171), (171, 49, 36)]
    text_list = text.split()
    text_list = [text for text in text_list if not has_numbers(text)]
    start = random.randint(0, len(text_list)-10)
    word = text_list[start:start+100]

    x = 10
    y = 10
    word_num = 0
    for w in word:

        if x+font.getsize(w)[0]+font.getsize(' ')[0] > 246:
            x = 10
            y = y+30

            if y+font.getsize(w)[1]>256:
                break

        color_seed = random.randint(1,20)
        if color_seed in [13,14]:
            c = color[1]
        elif color_seed == 15:
            c = color[2]
        else:
            c = color[0]

        imgDraw.text((x, y), w, font = font, fill=c)
        x = x+font.getsize(w)[0]+font.getsize(' ')[0]
        word_num = word_num+1

    words_dict[img_name] = word[:word_num]
    img.save("imgs/" + str(img_name) + ".png")
    
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    blur_rate = scale_size[font_seed]
    blur_img(open_cv_image,blur_rate, img_name)

if __name__ == '__main__':
    text_path = "/Users/jason543wu/Desktop/WikiResearch/shared/pair_data_organized_new/"
    words_dict = {}
    img_name = 1
    for fname in os.listdir(text_path)[:100]:
        with open(text_path + fname) as f:
            lines = f.readlines()
            if len(lines[0])<1000:
                continue
            for i in range(12):
                get_png(lines[0], img_name, words_dict)
                img_name += 1
                if img_name>1000:
                    break
        if img_name>1000:
            break

    with open("words_dict.pkl", "wb") as f:
        pickle.dump(words_dict, f)