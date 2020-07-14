import pickle
from joblib import dump, load
import cv2
import sys,os,pygame,random,math
import numpy as np
import math
import numpy as np
import os
import glob
import warnings
from skimage.feature import hog

threshold = 60  #  BINARY threshold
blurValue = 41


font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255,255,255) 
  
# Line thickness of 2 px 
thickness = 2

#load model
# clf = load('weight5.joblib') 
clf = load('weight_scale.joblib') 

camera = cv2.VideoCapture(0)
import time   

bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)



# def extract_feature(img):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) <1:
#         return False
#     ac=np.hstack([contours[i].flatten() for i in range(len(contours))])
#     l=ac.shape[0]
#     if l<10:
#         return False
#     else:
#         ac=ac.reshape(int(l/2),2)
#         (x,y),(MA,ma),angle = cv2.fitEllipse(ac)
#         area = cv2.contourArea(ac)
#         # Tính diện tích bouding box
#         x,y,w,h = cv2.boundingRect(ac)
#         rect_area = w*h
#         # Tính độ phủ
#         extent = float(area)/rect_area
#         H = hog(img, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
#         return np.hstack([angle,rect_area,H])




def extract_feature(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) <1:
        return False
    ac=np.hstack([contours[i].flatten() for i in range(len(contours))])
    l=ac.shape[0]
    if l<10:
        return False
    else:
        ac=ac.reshape(int(l/2),2)
        (x,y),(MA,ma),angle = cv2.fitEllipse(ac)
        area = cv2.contourArea(ac)
        # Tính diện tích bouding box
        x,y,w,h = cv2.boundingRect(ac)
        rect_area = w*h
        # Tính độ phủ
        extent = float(area)/rect_area
        H = hog(img, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(2,2), transform_sqrt=True, block_norm="L1")
        return np.hstack([(angle/180),extent,H])

def bgSubMasking(self, frame):
    """Create a foreground (hand) mask
    @param frame: The video frame
    @return: A masked frame
    """
    fgmask = bgSubtractor.apply(frame, learningRate=0)    

    kernel = np.ones((4, 4), np.uint8)
    
    # The effect is to remove the noise in the background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    # To close the holes in the objects
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply the mask on the frame and return
    return cv2.bitwise_and(frame, frame, mask=fgmask)

def histMasking(frame, handHist):
    """Create the HSV masking
    @param frame: The video frame
    @param handHist: The histogram generated
    @return: A masked frame
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], handHist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cv2.filter2D(dst, -1, disc, dst)
    # dst is now a probability map
    # Use binary thresholding to create a map of 0s and 1s
    # 1 means the pixel is part of the hand and 0 means not
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=7)
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(frame, thresh)   


pygame.init()
pygame.display.set_caption("Nsnake v1.0")
pygame.font.init()
random.seed()

#Global constant definitions
SPEED = 0.36
SNAKE_SIZE = 9
APPLE_SIZE = SNAKE_SIZE
SEPARATION = 10
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
FPS = 25
KEY = {"UP":1,"DOWN":2,"LEFT":3,"RIGHT":4}
#Screen initialization
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT),pygame.HWSURFACE)
# set where the display will move to
# x=200
# y=200
# os.environ['SDL_VIDEO_WINDOW_POS']='%d,%d' %(x,y)

#resize the screen causing it to move to x y set by environ
pygame.display.set_mode((501,200))

#set the size back to normal
pygame.display.set_mode((800,600))

#Resources
score_font = pygame.font.Font(None,38)
score_numb_font = pygame.font.Font(None,28)
game_over_font = pygame.font.Font(None,46)
play_again_font = score_numb_font
score_msg = score_font.render("Score:",1,pygame.Color("green"))
score_msg_size = score_font.size("Score")
#icon = pygame.image.load("nsnake32.png")
#pygame.display.set_icon(icon)

background_color = pygame.Color(74,74,74)
black = pygame.Color(0,0,0)

#Clock
gameClock = pygame.time.Clock()

def checkCollision(posA,As,posB,Bs):
    #As size of a | Bs size of B
    if(posA.x   < posB.x+Bs and posA.x+As > posB.x and posA.y < posB.y + Bs and posA.y+As > posB.y):
        return True
    return False

def checkLimits(entity):
    if(entity.x > SCREEN_WIDTH):
        entity.x = SNAKE_SIZE
    if(entity.x < 0):
        entity.x = SCREEN_WIDTH - SNAKE_SIZE
    if(entity.y > SCREEN_HEIGHT):
        entity.y = SNAKE_SIZE
    if(entity.y < 0):
        entity.y = SCREEN_HEIGHT - SNAKE_SIZE
        
class Apple:
    def __init__(self,x,y,state):
        self.x = x
        self.y = y
        self.state = state
        self.color = pygame.color.Color("red")
    def draw(self,screen):
        pygame.draw.rect(screen,self.color,(self.x,self.y,APPLE_SIZE,APPLE_SIZE),0)
        
class Segment:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.direction = KEY["UP"]
        self.color = "white"
        
class Snake:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.direction = KEY["UP"]
        self.stack = []
        
        self.stack.append(self)
        
        blackBox = Segment(self.x,self.y + SEPARATION)
        blackBox.direction = KEY["UP"]
        blackBox.color = "NULL"
        self.stack.append(blackBox)

        
        
    def move(self):
        last_element = len(self.stack)-1
        while(last_element != 0):
            self.stack[last_element].direction = self.stack[last_element-1].direction
            self.stack[last_element].x = self.stack[last_element-1].x 
            self.stack[last_element].y = self.stack[last_element-1].y 
            last_element-=1
        if(len(self.stack)<2):
            last_segment = self
        else:
            last_segment = self.stack.pop(last_element)
        last_segment.direction = self.stack[0].direction
        if(self.stack[0].direction ==KEY["UP"]):
            last_segment.y = self.stack[0].y - (SPEED * FPS)
        elif(self.stack[0].direction == KEY["DOWN"]):
            last_segment.y = self.stack[0].y + (SPEED * FPS) 
        elif(self.stack[0].direction ==KEY["LEFT"]):
            last_segment.x = self.stack[0].x - (SPEED * FPS)
        elif(self.stack[0].direction == KEY["RIGHT"]):
            last_segment.x = self.stack[0].x + (SPEED * FPS)
        self.stack.insert(0,last_segment)

    def getHead(self):
        return(self.stack[0])
    
    def grow(self):
        last_element = len(self.stack)-1
        self.stack[last_element].direction = self.stack[last_element].direction
        if(self.stack[last_element].direction == KEY["UP"]):
            newSegment = Segment(self.stack[last_element].x,self.stack[last_element].y-SNAKE_SIZE)
            blackBox = Segment(newSegment.x,newSegment.y-SEPARATION)
            
        elif(self.stack[last_element].direction == KEY["DOWN"]):
            newSegment = Segment(self.stack[last_element].x,self.stack[last_element].y+SNAKE_SIZE)
            blackBox = Segment(newSegment.x,newSegment.y+SEPARATION)
            
        elif(self.stack[last_element].direction == KEY["LEFT"]):
            newSegment = Segment(self.stack[last_element].x-SNAKE_SIZE,self.stack[last_element].y)
            blackBox = Segment(newSegment.x-SEPARATION,newSegment.y)
            
        elif(self.stack[last_element].direction == KEY["RIGHT"]):
            newSegment = Segment(self.stack[last_element].x+SNAKE_SIZE,self.stack[last_element].y)
            blackBox = Segment(newSegment.x+SEPARATION,newSegment.y)
            
        blackBox.color = "NULL"
        self.stack.append(newSegment)
        self.stack.append(blackBox)
        
    def iterateSegments(self,delta):
        pass
    
    def setDirection(self,direction):
        if(self.direction == KEY["RIGHT"] and direction == KEY["LEFT"] or self.direction == KEY["LEFT"] and direction == KEY["RIGHT"]):
            pass
        elif(self.direction == KEY["UP"] and direction == KEY["DOWN"] or self.direction == KEY["DOWN"] and direction == KEY["UP"]):
            pass
        else:
            self.direction = direction
            
    def get_rect(self):
        rect = (self.x,self.y)
        return rect
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def setX(self,x):
        self.x = x
        
    def setY(self,y):
        self.y = y
        
    def checkCrash(self):
        counter = 1
        while(counter < len(self.stack)-1):
            if(checkCollision(self.stack[0],SNAKE_SIZE,self.stack[counter],SNAKE_SIZE)and self.stack[counter].color != "NULL"):
                return True
            counter+=1
        return False
    
    def draw(self,screen):
        pygame.draw.rect(screen,pygame.color.Color("yellow"),(self.stack[0].x,self.stack[0].y,SNAKE_SIZE,SNAKE_SIZE),0)
        counter = 1
        while(counter < len(self.stack)):
            if(self.stack[counter].color == "NULL"):
                counter+=1
                continue
            pygame.draw.rect(screen,pygame.color.Color("white"),(self.stack[counter].x,self.stack[counter].y,SNAKE_SIZE,SNAKE_SIZE),0)
            counter+=1
        
                
def getKey():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return KEY["UP"]
                elif event.key == pygame.K_DOWN:
                    return KEY["DOWN"]
                elif event.key == pygame.K_LEFT:
                    return KEY["LEFT"]
                elif event.key == pygame.K_RIGHT:
                    return KEY["RIGHT"]
                elif event.key == pygame.K_ESCAPE:
                    return "exit"
                elif event.key == pygame.K_y:
                    return "yes"
                elif event.key == pygame.K_n:
                    return "no"
            if event.type == pygame.QUIT:
                sys.exit()

def respawnApple(apples,index,sx,sy):
    radius = math.sqrt((SCREEN_WIDTH/2*SCREEN_WIDTH/2  + SCREEN_HEIGHT/2*SCREEN_HEIGHT/2))/2
    angle = 999
    while(angle > radius):
        angle = random.uniform(0,800)*math.pi*2
        x = SCREEN_WIDTH/2 + radius * math.cos(angle)
        y = SCREEN_HEIGHT/2 + radius * math.sin(angle)
        if(x == sx and y == sy):
            continue
    newApple = Apple(x,y,1)
    apples[index] = newApple
        
def respawnApples(apples,quantity,sx,sy):
    counter = 0
    del apples[:]
    radius = math.sqrt((SCREEN_WIDTH/2*SCREEN_WIDTH/2  + SCREEN_HEIGHT/2*SCREEN_HEIGHT/2))/2
    angle = 999
    while(counter < quantity):
        while(angle > radius):
            angle = random.uniform(0,800)*math.pi*2
            x = SCREEN_WIDTH/2 + radius * math.cos(angle)
            y = SCREEN_HEIGHT/2 + radius * math.sin(angle)
            if( (x-APPLE_SIZE == sx or x+APPLE_SIZE == sx) and (y-APPLE_SIZE == sy or y+APPLE_SIZE == sy) or radius - angle <= 10):
                continue
        apples.append(Apple(x,y,1))
        angle = 999
        counter+=1
        
def endGame():
    message = game_over_font.render("Game Over",1,pygame.Color("white"))
    message_play_again = play_again_font.render("Play Again? Y/N",1,pygame.Color("green"))
    screen.blit(message,(320,240))
    screen.blit(message_play_again,(320+12,240+40))

    pygame.display.flip()
    pygame.display.update()
    
    myKey = getKey()
    while(myKey != "exit"):
        if(myKey == "yes"):
            main()
        elif(myKey == "no"):
            break
        myKey = getKey()
        gameClock.tick(FPS)
    sys.exit()

def drawScore(score):
    score_numb = score_numb_font.render(str(score),1,pygame.Color("red"))
    screen.blit(score_msg, (SCREEN_WIDTH-score_msg_size[0]-60,10) )
    screen.blit(score_numb,(SCREEN_WIDTH - 45,14))
    
def drawGameTime(gameTime):
    game_time = score_font.render("Time:",1,pygame.Color("green"))
    game_time_numb = score_numb_font.render(str(gameTime/1000),1,pygame.Color("red"))
    screen.blit(game_time,(30,10))
    screen.blit(game_time_numb,(105,14))
    
def exitScreen():
    pass



def bgSubMasking(self, frame):
    """Create a foreground (hand) mask
    @param frame: The video frame
    @return: A masked frame
    """
    fgmask = bgSubtractor.apply(frame, learningRate=0)    

    kernel = np.ones((4, 4), np.uint8)
    
    # The effect is to remove the noise in the background
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    # To close the holes in the objects
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply the mask on the frame and return
    return cv2.bitwise_and(frame, frame, mask=fgmask)


def getLabel(id):
    y=None
    if id==0:
        y='DOWN'
    elif id==1:
        y='LEFT'
    elif id==2:
        y='RIGHT'
    else:
        y='UP'
    return y
    

def convertKey(id):
    if id==0:
        return 2
    elif id==1:
        return 3
    elif id==2:
        return 4
    else:
        return 1

time.sleep(2)
def main():
    score = 0

    # Snake initialization
    mySnake = Snake(SCREEN_WIDTH/2,SCREEN_HEIGHT/2)
    mySnake.setDirection(KEY["UP"])
    mySnake.move()
    start_segments=3
    while(start_segments>0):
        mySnake.grow()
        mySnake.move() 
        start_segments-=1

    #Apples
    max_apples = 1
    eaten_apple = False
    apples = [Apple(random.randint(60,SCREEN_WIDTH),random.randint(60,SCREEN_HEIGHT),1)]
    respawnApples(apples,max_apples,mySnake.x,mySnake.y)
    
    startTime = pygame.time.get_ticks()
    endgame = 0
    
    while(camera.isOpened() and endgame!=1):
            ret, frame = camera.read()
            frame=cv2.flip(frame,1)
            frame=frame[100:450,400:700]
            frame = cv2.resize(frame, fixed_size)
            #background subtractor
            fram=bgSubMasking(frame,frame)
            gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            features=[]
            feature=extract_feature(thresh)
            if type(feature)!=bool:
                features.append(feature)
                y_predict=clf.predict(features)
                id=getLabel(y_predict)
                fram = cv2.putText(thresh, id, org, font, fontScale, color, thickness, cv2.LINE_AA) 
                keyPress=convertKey(y_predict)        
                cv2.imshow('img',fram)
                cv2.moveWindow('img',1200,100)
                cv2.imshow('original',frame)
                cv2.moveWindow('original',1200,450)
                gameClock.tick(FPS)

                if keyPress == "exit":
                    endgame = 1
            
                #Collision check
                checkLimits(mySnake)
                if(mySnake.checkCrash()== True):
                    endGame()
                    
                for myApple in apples:
                    if(myApple.state == 1):
                        if(checkCollision(mySnake.getHead(),SNAKE_SIZE,myApple,APPLE_SIZE)==True):
                            mySnake.grow()
                            myApple.state = 0
                            score+=5
                            eaten_apple=True
                    

                #Position Update
                if(keyPress):
                    mySnake.setDirection(keyPress)    
                mySnake.move()
                
                
                
                #Respawning apples
                if(eaten_apple == True):
                    eaten_apple = False
                    respawnApple(apples,0,mySnake.getHead().x,mySnake.getHead().y)

                #Drawing
                screen.fill(background_color)
                for myApple in apples:
                    if(myApple.state == 1):
                        myApple.draw(screen)
                        
                mySnake.draw(screen)
                drawScore(score)
                gameTime = pygame.time.get_ticks() - startTime
                drawGameTime(gameTime)
                
                pygame.display.flip()
                pygame.display.update()
                if cv2.waitKey(25) & 0xFF == ord('q'):
                            break


fixed_size=(320,320)
main()
camera.release()
cv2.destroyAllWindows()


