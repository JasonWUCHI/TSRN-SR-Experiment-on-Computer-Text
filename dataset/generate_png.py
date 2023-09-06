from PIL import Image, ImageDraw, ImageFont
import os, random

def get_png(text, img_name):
    width = 256
    height = 256
    img = Image.new('RGB', (width, height), color='white')
    font = ImageFont.truetype("arial.ttf", size=20)

    imgDraw = ImageDraw.Draw(img)

    y = 10
    color = [(0,0,0), (36,25,171), (171, 49, 36)]
    for i in range(8):
        start = random.randint(0, len(text)-100)
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