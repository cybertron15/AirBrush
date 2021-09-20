import cv2 as cv
import os

def masking(UI_element):
    """Creates masks for background and UI componenet"""
    gray = cv.cvtColor(UI_element,cv.COLOR_BGR2GRAY)
    t, backgroond_mask = cv.threshold(gray,150,255,cv.THRESH_BINARY)
    t, UI_element_mask = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
    masked_UI_element = cv.bitwise_and(UI_element,UI_element,mask=UI_element_mask)
    
    return masked_UI_element,backgroond_mask

def load_UI_components(component_folder_path):
    """Takes the folder path of all the UI componenets, loads all of the
    UI componenets and maps the UI componenets attribute
    to the UI componenets name in a dictionary"""

    UI_componenets_paths = os.listdir(component_folder_path)
    UI_componenets_positions = [[5,150],[5,150],[5,210],[5,210],[0,0],[5,270],[5,270]]
    resizeable_components = [(),(),(),(),(640,40),(),()]

    UI_components = {}
    for index, paths in enumerate(UI_componenets_paths):
        component = cv.imread(r"{}/{}".format(component_folder_path,paths))
        position = UI_componenets_positions[index]
        resizable = resizeable_components[index]
        if resizable:
            component = cv.resize(component,resizable)
        masked_UI_element,backgroond_mask = masking(component)
        UI_name = paths[:-4]
        UI_components[UI_name]=[component,position,resizable,masked_UI_element,backgroond_mask]

    return UI_components