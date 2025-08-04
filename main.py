from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from photo_geo import CameraScreen, GalleryScreen, DiaryScreen, SettingsScreen, MapScreen

class PhotoGeoApp(App):
    def build(self):
        # Создаем менеджер экранов
        sm = ScreenManager()
        
        # Добавляем экраны
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(GalleryScreen(name='gallery'))
        sm.add_widget(DiaryScreen(name='diary'))
        sm.add_widget(SettingsScreen(name='settings'))
        sm.add_widget(MapScreen(name='map'))
        
        return sm

if __name__ == '__main__':
    PhotoGeoApp().run()