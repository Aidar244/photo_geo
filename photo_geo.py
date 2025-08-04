import os
import json
import uuid
from datetime import datetime
from functools import partial
import cv2
from PIL import Image, ImageDraw, ImageFont
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.camera import Camera
from kivy.uix.image import Image as KivyImage
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty
from kivy.utils import platform
from plyer import gps, camera, storagepath
from geopy.geocoders import Nominatim
import numpy as np

# Модели данных
class PhotoData:
    def __init__(self, latitude, longitude, altitude, accuracy, timestamp, custom_text="", template_text="", watermarks=None):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.accuracy = accuracy
        self.timestamp = timestamp
        self.custom_text = custom_text
        self.template_text = template_text
        self.watermarks = watermarks or ["PhotoGeo", "Protected"]

    def to_overlay_text(self):
        coords = f"{self.latitude:.6f}, {self.longitude:.6f}"
        accuracy_text = f"(±{self.accuracy:.1f}m)"
        altitude_text = f", {self.altitude:.1f}m" if self.altitude else ""
        date_text = self.timestamp.strftime("%Y-%m-%d")
        
        result = f"{coords} {accuracy_text}{altitude_text}\n{date_text}"
        
        if self.template_text:
            result = f"{result}\n{self.template_text}"
            
        if self.custom_text:
            result = f"{result}\n{self.custom_text}"
            
        return result

class DiaryEntry:
    def __init__(self, title, content, created_at=None, updated_at=None, is_encrypted=False):
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.is_encrypted = is_encrypted

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_encrypted': self.is_encrypted
        }

    @classmethod
    def from_dict(cls, data):
        entry = cls(
            title=data['title'],
            content=data['content'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            is_encrypted=data.get('is_encrypted', False)
        )
        entry.id = data['id']
        return entry

class TemplateText:
    def __init__(self, name, content, created_at=None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.content = content
        self.created_at = created_at or datetime.now()

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'content': self.content,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data):
        template = cls(
            name=data['name'],
            content=data['content'],
            created_at=datetime.fromisoformat(data['created_at'])
        )
        template.id = data['id']
        return template

# Сервисы
class LocationService:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.accuracy = 0.0
        self.is_available = False

    def start_gps(self):
        try:
            gps.configure(on_location=self.on_location)
            gps.start(1000, 0)  # Обновление каждую секунду
            return True
        except Exception as e:
            print(f"Ошибка запуска GPS: {e}")
            return False

    def stop_gps(self):
        try:
            gps.stop()
        except Exception as e:
            print(f"Ошибка остановки GPS: {e}")

    def on_location(self, **kwargs):
        self.latitude = kwargs.get('lat', 0.0)
        self.longitude = kwargs.get('lon', 0.0)
        self.altitude = kwargs.get('altitude', 0.0)
        self.accuracy = kwargs.get('accuracy', 0.0)
        self.is_available = True

    def get_current_location(self):
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy': self.accuracy,
            'is_available': self.is_available
        }

class StorageService:
    def __init__(self):
        self.app_dir = self._get_app_directory()
        self.photos_dir = os.path.join(self.app_dir, "photos")
        self.diary_file = os.path.join(self.app_dir, "diary.json")
        self.templates_file = os.path.join(self.app_dir, "templates.json")
        self.settings_file = os.path.join(self.app_dir, "settings.json")
        
        # Создаем директории
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.diary_file), exist_ok=True)

    def _get_app_directory(self):
        if platform == 'android':
            from jnius import autoclass
            Environment = autoclass('android.os.Environment')
            return os.path.join(
                Environment.getExternalStorageDirectory().getAbsolutePath(),
                'PhotoGeo'
            )
        else:
            return os.path.join(os.path.expanduser("~"), ".photo_geo")

    def save_photo(self, image_array, photo_data):
        """Сохраняет фото с наложенными данными"""
        try:
            # Создаем изображение PIL из массива numpy
            image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            
            # Добавляем вотермарки
            self._add_watermarks(image, photo_data.watermarks)
            
            # Добавляем текстовые данные
            self._add_overlay_text(image, photo_data)
            
            # Сохраняем фото
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_geo_{timestamp}.png"
            filepath = os.path.join(self.photos_dir, filename)
            image.save(filepath, "PNG")
            
            # Добавляем EXIF данные
            self._add_exif_data(filepath, photo_data)
            
            return filepath
        except Exception as e:
            print(f"Ошибка сохранения фото: {e}")
            return None

    def _add_watermarks(self, image, watermarks):
        """Добавляет вотермарки на изображение"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        try:
            # Используем стандартный шрифт
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Добавляем вотермарки в верхний левый угол
        for i, watermark in enumerate(watermarks[:2]):  # Только первые две
            text_position = (20, 20 + i * 30)
            draw.text(text_position, watermark, fill=(255, 255, 255, 128), font=font)

    def _add_overlay_text(self, image, photo_data):
        """Добавляет текст с данными на изображение"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        try:
            # Используем стандартный шрифт
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Текст для наложения
        overlay_text = photo_data.to_overlay_text()
        
        # Рисуем полупрозрачный фон для текста
        lines = overlay_text.split('\n')
        text_height = len(lines) * 20
        background_position = [(10, height - text_height - 20), 
                              (width - 10, height - 10)]
        draw.rectangle(background_position, fill=(0, 0, 0, 128))
        
        # Рисуем текст
        for i, line in enumerate(lines):
            text_position = (20, height - text_height - 10 + i * 20)
            draw.text(text_position, line, fill=(255, 255, 255), font=font)

    def _add_exif_data(self, filepath, photo_data):
        """Добавляет EXIF данные в изображение"""
        # В этой реализации мы просто сохраняем данные в отдельный файл
        # Для полноценной работы с EXIF потребуется дополнительная библиотека
        exif_data = {
            'GPSLatitude': photo_data.latitude,
            'GPSLongitude': photo_data.longitude,
            'GPSAltitude': photo_data.altitude,
            'DateTime': photo_data.timestamp.isoformat()
        }
        
        exif_file = filepath.replace('.png', '_exif.json')
        with open(exif_file, 'w') as f:
            json.dump(exif_data, f)

    def get_photos(self):
        """Получает список сохраненных фото"""
        photos = []
        if os.path.exists(self.photos_dir):
            for filename in os.listdir(self.photos_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    photos.append(os.path.join(self.photos_dir, filename))
        return sorted(photos, reverse=True)

    def get_diary_entries(self):
        """Получает записи дневника"""
        if not os.path.exists(self.diary_file):
            return []
        
        try:
            with open(self.diary_file, 'r') as f:
                data = json.load(f)
                return [DiaryEntry.from_dict(entry) for entry in data]
        except Exception as e:
            print(f"Ошибка загрузки записей дневника: {e}")
            return []

    def save_diary_entry(self, entry):
        """Сохраняет запись дневника"""
        entries = self.get_diary_entries()
        
        # Проверяем, существует ли запись
        existing_index = None
        for i, existing_entry in enumerate(entries):
            if existing_entry.id == entry.id:
                existing_index = i
                break
        
        if existing_index is not None:
            entries[existing_index] = entry
        else:
            entries.append(entry)
        
        # Сохраняем все записи
        try:
            with open(self.diary_file, 'w') as f:
                json.dump([entry.to_dict() for entry in entries], f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения записи дневника: {e}")

    def delete_diary_entry(self, entry_id):
        """Удаляет запись дневника"""
        entries = self.get_diary_entries()
        entries = [entry for entry in entries if entry.id != entry_id]
        
        try:
            with open(self.diary_file, 'w') as f:
                json.dump([entry.to_dict() for entry in entries], f, indent=2)
        except Exception as e:
            print(f"Ошибка удаления записи дневника: {e}")

    def get_text_templates(self):
        """Получает шаблоны текста"""
        if not os.path.exists(self.templates_file):
            return []
        
        try:
            with open(self.templates_file, 'r') as f:
                data = json.load(f)
                return [TemplateText.from_dict(template) for template in data]
        except Exception as e:
            print(f"Ошибка загрузки шаблонов: {e}")
            return []

    def save_text_template(self, template):
        """Сохраняет шаблон текста"""
        templates = self.get_text_templates()
        
        # Ограничиваем до 5 шаблонов
        if len(templates) >= 5 and not any(t.id == template.id for t in templates):
            templates = templates[1:]  # Удаляем самый старый
        
        # Проверяем, существует ли шаблон
        existing_index = None
        for i, existing_template in enumerate(templates):
            if existing_template.id == template.id:
                existing_index = i
                break
        
        if existing_index is not None:
            templates[existing_index] = template
        else:
            templates.append(template)
        
        # Сохраняем все шаблоны
        try:
            with open(self.templates_file, 'w') as f:
                json.dump([template.to_dict() for template in templates], f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения шаблона: {e}")

    def delete_text_template(self, template_id):
        """Удаляет шаблон текста"""
        templates = self.get_text_templates()
        templates = [template for template in templates if template.id != template_id]
        
        try:
            with open(self.templates_file, 'w') as f:
                json.dump([template.to_dict() for template in templates], f, indent=2)
        except Exception as e:
            print(f"Ошибка удаления шаблона: {e}")

    def get_settings(self):
        """Получает настройки приложения"""
        if not os.path.exists(self.settings_file):
            return {}
        
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки настроек: {e}")
            return {}

    def save_settings(self, settings):
        """Сохраняет настройки приложения"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")

# Основные экраны приложения
class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location_service = LocationService()
        self.storage_service = StorageService()
        self.photo_count = 0
        self.is_quick_mode = False
        self.selected_template = None
        self.custom_text = ""
        
        # Запускаем GPS
        self.location_service.start_gps()
        
        # Создаем интерфейс
        self.create_ui()
        
        # Обновляем информацию о местоположении
        Clock.schedule_interval(self.update_location_info, 1)

    def create_ui(self):
        """Создает пользовательский интерфейс"""
        layout = FloatLayout()
        
        # Камера
        self.camera = Camera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)
        
        # Панель информации в верхней части
        info_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=50,
            pos_hint={'top': 1}
        )
        
        # Индикатор местоположения
        self.location_label = Label(
            text="Поиск местоположения...",
            color=(1, 1, 1, 1),
            size_hint_x=None,
            width=300
        )
        info_layout.add_widget(self.location_label)
        
        # Счетчик фото
        self.counter_label = Label(
            text=f"Фото: {self.photo_count}",
            color=(1, 1, 1, 1),
            size_hint_x=None,
            width=100
        )
        info_layout.add_widget(self.counter_label)
        
        # Кнопки навигации
        nav_layout = BoxLayout(orientation='horizontal', size_hint_x=None, width=200)
        
        gallery_btn = Button(text="Галерея", size_hint_x=None, width=80)
        gallery_btn.bind(on_press=self.go_to_gallery)
        nav_layout.add_widget(gallery_btn)
        
        diary_btn = Button(text="Дневник", size_hint_x=None, width=80)
        diary_btn.bind(on_press=self.go_to_diary)
        nav_layout.add_widget(diary_btn)
        
        settings_btn = Button(text="Настройки", size_hint_x=None, width=80)
        settings_btn.bind(on_press=self.go_to_settings)
        nav_layout.add_widget(settings_btn)
        
        info_layout.add_widget(nav_layout)
        layout.add_widget(info_layout)
        
        # Панель ввода текста в нижней части
        text_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(1, None),
            height=50,
            pos_hint={'bottom': 1}
        )
        
        # Выбор шаблона
        self.template_btn = Button(
            text="Шаблон",
            size_hint_x=None,
            width=100
        )
        self.template_btn.bind(on_press=self.show_template_selector)
        text_layout.add_widget(self.template_btn)
        
        # Поле ввода текста
        self.text_input = TextInput(
            hint_text="Добавить заметку...",
            multiline=False,
            size_hint_x=0.7
        )
        self.text_input.bind(text=self.on_text_change)
        text_layout.add_widget(self.text_input)
        
        # Кнопка быстрого режима
        self.quick_mode_btn = ToggleButton(
            text="Быстрый режим",
            size_hint_x=None,
            width=120
        )
        self.quick_mode_btn.bind(on_press=self.toggle_quick_mode)
        text_layout.add_widget(self.quick_mode_btn)
        
        layout.add_widget(text_layout)
        
        # Кнопка съемки
        capture_btn = Button(
            text="Снять фото",
            size_hint=(None, None),
            size=(80, 80),
            pos_hint={'center_x': 0.5, 'y': 0.1}
        )
        capture_btn.bind(on_press=self.capture_photo)
        layout.add_widget(capture_btn)
        
        self.add_widget(layout)

    def update_location_info(self, dt):
        """Обновляет информацию о местоположении"""
        location = self.location_service.get_current_location()
        
        if not location['is_available']:
            self.location_label.text = "GPS недоступен"
            self.location_label.color = (1, 0, 0, 1)  # Красный
        else:
            lat = location['latitude']
            lon = location['longitude']
            accuracy = location['accuracy']
            self.location_label.text = f"{lat:.6f}, {lon:.6f} (±{accuracy:.1f}м)"
            self.location_label.color = (1, 1, 1, 1)  # Белый

    def on_text_change(self, instance, value):
        """Обработчик изменения текста"""
        self.custom_text = value

    def show_template_selector(self, instance):
        """Показывает выбор шаблона текста"""
        templates = self.storage_service.get_text_templates()
        
        if not templates:
            popup = Popup(
                title="Шаблоны текста",
                content=Label(text="Нет доступных шаблонов"),
                size_hint=(0.8, 0.6)
            )
            popup.open()
            return
        
        # Создаем список шаблонов
        layout = BoxLayout(orientation='vertical')
        
        for template in templates:
            btn = Button(
                text=template.name,
                size_hint_y=None,
                height=40
            )
            btn.bind(on_press=partial(self.select_template, template))
            layout.add_widget(btn)
        
        close_btn = Button(
            text="Закрыть",
            size_hint_y=None,
            height=40
        )
        
        popup_layout = BoxLayout(orientation='vertical')
        popup_layout.add_widget(ScrollView(size_hint=(1, 0.9)))
        popup_layout.children[0].add_widget(layout)
        popup_layout.add_widget(close_btn)
        
        popup = Popup(
            title="Выберите шаблон",
            content=popup_layout,
            size_hint=(0.8, 0.8)
        )
        
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def select_template(self, template, instance):
        """Выбирает шаблон текста"""
        self.selected_template = template
        self.template_btn.text = template.name[:10] + "..." if len(template.name) > 10 else template.name

    def toggle_quick_mode(self, instance):
        """Переключает быстрый режим"""
        self.is_quick_mode = instance.state == 'down'

    def capture_photo(self, instance):
        """Делает снимок"""
        # Получаем кадр с камеры
        texture = self.camera.texture
        if texture is None:
            return
            
        # Преобразуем текстуру в массив numpy
        pixels = texture.pixels
        size = texture.size
        fmt = texture.colorfmt
        
        # Создаем массив numpy
        image_array = np.frombuffer(pixels, dtype=np.uint8)
        image_array = image_array.reshape((size[1], size[0], 4))  # RGBA
        
        # Преобразуем RGBA в BGR
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        
        # Получаем данные о местоположении
        location = self.location_service.get_current_location()
        if not location['is_available']:
            # Показываем сообщение об ошибке
            popup = Popup(
                title="Ошибка",
                content=Label(text="Местоположение недоступно"),
                size_hint=(0.6, 0.4)
            )
            popup.open()
            return
        
        # Создаем объект с данными фото
        photo_data = PhotoData(
            latitude=location['latitude'],
            longitude=location['longitude'],
            altitude=location['altitude'],
            accuracy=location['accuracy'],
            timestamp=datetime.now(),
            custom_text=self.custom_text,
            template_text=self.selected_template.content if self.selected_template else "",
            watermarks=["PhotoGeo", "Protected"]
        )
        
        # Сохраняем фото
        filepath = self.storage_service.save_photo(image_array, photo_data)
        
        if filepath:
            self.photo_count += 1
            self.counter_label.text = f"Фото: {self.photo_count}"
            
            # Если включен быстрый режим, показываем уведомление
            if self.is_quick_mode:
                popup = Popup(
                    title="Успех",
                    content=Label(text="Фото сохранено!"),
                    size_hint=(0.6, 0.4)
                )
                popup.open()
                Clock.schedule_once(lambda dt: popup.dismiss(), 2)
            else:
                # В обычном режиме открываем редактор (упрощенная версия)
                self.show_editor(filepath)
        else:
            popup = Popup(
                title="Ошибка",
                content=Label(text="Не удалось сохранить фото"),
                size_hint=(0.6, 0.4)
            )
            popup.open()

    def show_editor(self, filepath):
        """Показывает редактор фото (упрощенная версия)"""
        # В этой версии просто показываем сообщение
        popup = Popup(
            title="Редактор",
            content=Label(text="Редактор фото (в разработке)"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def go_to_gallery(self, instance):
        """Переходит к галерее"""
        app = App.get_running_app()
        app.root.current = 'gallery'

    def go_to_diary(self, instance):
        """Переходит к дневнику"""
        app = App.get_running_app()
        app.root.current = 'diary'

    def go_to_settings(self, instance):
        """Переходит к настройкам"""
        app = App.get_running_app()
        app.root.current = 'settings'

    def on_enter(self):
        """Вызывается при переходе на экран"""
        # Возобновляем работу камеры
        if hasattr(self, 'camera'):
            self.camera.play = True

    def on_leave(self):
        """Вызывается при уходе с экрана"""
        # Останавливаем камеру
        if hasattr(self, 'camera'):
            self.camera.play = False

class GalleryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage_service = StorageService()
        self.create_ui()

    def create_ui(self):
        """Создает интерфейс галереи"""
        layout = BoxLayout(orientation='vertical')
        
        # Заголовок
        header = BoxLayout(size_hint_y=None, height=50)
        header.add_widget(Label(text="Галерея PhotoGeo", size_hint_x=0.8))
        
        refresh_btn = Button(text="Обновить", size_hint_x=0.2)
        refresh_btn.bind(on_press=self.refresh_gallery)
        header.add_widget(refresh_btn)
        
        layout.add_widget(header)
        
        # Список фото
        self.scroll_view = ScrollView()
        self.photos_layout = GridLayout(cols=2, spacing=10, size_hint_y=None)
        self.photos_layout.bind(minimum_height=self.photos_layout.setter('height'))
        self.scroll_view.add_widget(self.photos_layout)
        layout.add_widget(self.scroll_view)
        
        # Кнопка возврата
        back_btn = Button(
            text="Назад",
            size_hint_y=None,
            height=50
        )
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
        
        # Загружаем фото
        self.refresh_gallery()

    def refresh_gallery(self, instance=None):
        """Обновляет список фото"""
        # Очищаем текущий список
        self.photos_layout.clear_widgets()
        
        # Получаем список фото
        photos = self.storage_service.get_photos()
        
        if not photos:
            self.photos_layout.add_widget(Label(text="Нет сохраненных фотографий"))
            return
        
        # Добавляем фото в сетку
        for photo_path in photos:
            # Создаем виджет для фото
            photo_widget = BoxLayout(orientation='vertical', size_hint_y=None, height=200)
            
            # Изображение
            try:
                # Для упрощения используем кнопку с текстом
                img_btn = Button(
                    text=os.path.basename(photo_path),
                    size_hint_y=0.8
                )
                img_btn.bind(on_press=partial(self.show_photo, photo_path))
                photo_widget.add_widget(img_btn)
            except Exception as e:
                print(f"Ошибка загрузки фото: {e}")
                photo_widget.add_widget(Label(text="Ошибка"))
            
            # Кнопки действий
            actions_layout = BoxLayout(size_hint_y=0.2)
            
            edit_btn = Button(text="Ред.", size_hint_x=0.3)
            edit_btn.bind(on_press=partial(self.edit_photo, photo_path))
            actions_layout.add_widget(edit_btn)
            
            share_btn = Button(text="Поделиться", size_hint_x=0.4)
            share_btn.bind(on_press=partial(self.share_photo, photo_path))
            actions_layout.add_widget(share_btn)
            
            delete_btn = Button(text="Удалить", size_hint_x=0.3)
            delete_btn.bind(on_press=partial(self.delete_photo, photo_path))
            actions_layout.add_widget(delete_btn)
            
            photo_widget.add_widget(actions_layout)
            self.photos_layout.add_widget(photo_widget)

    def show_photo(self, photo_path, instance):
        """Показывает фото в полном размере"""
        # В упрощенной версии просто показываем путь
        popup = Popup(
            title="Фото",
            content=Label(text=photo_path),
            size_hint=(0.8, 0.6)
        )
        popup.open()

    def edit_photo(self, photo_path, instance):
        """Редактирует фото"""
        popup = Popup(
            title="Редактирование",
            content=Label(text="Функция редактирования в разработке"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def share_photo(self, photo_path, instance):
        """Делится фото"""
        popup = Popup(
            title="Поделиться",
            content=Label(text=f"Поделиться: {photo_path}"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def delete_photo(self, photo_path, instance):
        """Удаляет фото"""
        def confirm_delete(*args):
            try:
                os.remove(photo_path)
                # Также удаляем файл EXIF, если он существует
                exif_file = photo_path.replace('.png', '_exif.json')
                if os.path.exists(exif_file):
                    os.remove(exif_file)
                self.refresh_gallery()
                confirm_popup.dismiss()
            except Exception as e:
                error_popup = Popup(
                    title="Ошибка",
                    content=Label(text=f"Не удалось удалить фото: {e}"),
                    size_hint=(0.6, 0.4)
                )
                error_popup.open()
        
        confirm_popup = Popup(
            title="Подтверждение",
            content=BoxLayout(orientation='vertical'),
            size_hint=(0.6, 0.4)
        )
        
        content = confirm_popup.content
        content.add_widget(Label(text="Удалить это фото?"))
        
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        cancel_btn = Button(text="Отмена")
        cancel_btn.bind(on_press=confirm_popup.dismiss)
        buttons_layout.add_widget(cancel_btn)
        
        delete_btn = Button(text="Удалить", background_color=(1, 0, 0, 1))
        delete_btn.bind(on_press=confirm_delete)
        buttons_layout.add_widget(delete_btn)
        
        content.add_widget(buttons_layout)
        confirm_popup.open()

    def go_back(self, instance):
        """Возвращается к камере"""
        app = App.get_running_app()
        app.root.current = 'camera'

class DiaryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage_service = StorageService()
        self.create_ui()

    def create_ui(self):
        """Создает интерфейс дневника"""
        layout = BoxLayout(orientation='vertical')
        
        # Заголовок
        header = BoxLayout(size_hint_y=None, height=50)
        header.add_widget(Label(text="Дневник", size_hint_x=0.8))
        
        add_btn = Button(text="Добавить", size_hint_x=0.2)
        add_btn.bind(on_press=self.add_entry)
        header.add_widget(add_btn)
        
        layout.add_widget(header)
        
        # Список записей
        self.scroll_view = ScrollView()
        self.entries_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.entries_layout.bind(minimum_height=self.entries_layout.setter('height'))
        self.scroll_view.add_widget(self.entries_layout)
        layout.add_widget(self.scroll_view)
        
        # Кнопка возврата
        back_btn = Button(
            text="Назад",
            size_hint_y=None,
            height=50
        )
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
        
        # Загружаем записи
        self.refresh_entries()

    def refresh_entries(self):
        """Обновляет список записей"""
        # Очищаем текущий список
        self.entries_layout.clear_widgets()
        
        # Получаем список записей
        entries = self.storage_service.get_diary_entries()
        
        if not entries:
            self.entries_layout.add_widget(Label(text="Нет записей в дневнике"))
            return
        
        # Добавляем записи
        for entry in entries:
            entry_widget = BoxLayout(orientation='vertical', size_hint_y=None, height=100)
            
            # Заголовок и дата
            title_layout = BoxLayout(size_hint_y=None, height=30)
            title_layout.add_widget(Label(text=entry.title, halign='left', valign='middle'))
            date_text = entry.updated_at.strftime("%d.%m.%Y")
            title_layout.add_widget(Label(text=date_text, halign='right', valign='middle'))
            entry_widget.add_widget(title_layout)
            
            # Краткое содержание
            content_preview = entry.content[:100] + "..." if len(entry.content) > 100 else entry.content
            entry_widget.add_widget(Label(text=content_preview, halign='left', valign='top'))
            
            # Кнопки действий
            actions_layout = BoxLayout(size_hint_y=None, height=40)
            
            edit_btn = Button(text="Редактировать", size_hint_x=0.5)
            edit_btn.bind(on_press=partial(self.edit_entry, entry))
            actions_layout.add_widget(edit_btn)
            
            delete_btn = Button(text="Удалить", size_hint_x=0.5, background_color=(1, 0, 0, 1))
            delete_btn.bind(on_press=partial(self.delete_entry, entry))
            actions_layout.add_widget(delete_btn)
            
            entry_widget.add_widget(actions_layout)
            self.entries_layout.add_widget(entry_widget)

    def add_entry(self, instance):
        """Добавляет новую запись"""
        entry = DiaryEntry("Новая запись", "")
        self.edit_entry(entry, instance)

    def edit_entry(self, entry, instance):
        """Редактирует запись"""
        # Создаем попап для редактирования
        popup = Popup(
            title="Редактировать запись",
            size_hint=(0.9, 0.9)
        )
        
        layout = BoxLayout(orientation='vertical')
        
        # Заголовок
        title_input = TextInput(
            text=entry.title,
            hint_text="Заголовок",
            size_hint_y=None,
            height=50
        )
        layout.add_widget(title_input)
        
        # Содержание
        content_input = TextInput(
            text=entry.content,
            hint_text="Содержание",
            multiline=True
        )
        layout.add_widget(content_input)
        
        # Кнопки
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        
        cancel_btn = Button(text="Отмена")
        cancel_btn.bind(on_press=popup.dismiss)
        buttons_layout.add_widget(cancel_btn)
        
        save_btn = Button(text="Сохранить")
        save_btn.bind(on_press=partial(self.save_entry, entry, title_input, content_input, popup))
        buttons_layout.add_widget(save_btn)
        
        layout.add_widget(buttons_layout)
        popup.content = layout
        popup.open()

    def save_entry(self, entry, title_input, content_input, popup, instance):
        """Сохраняет запись"""
        entry.title = title_input.text
        entry.content = content_input.text
        entry.updated_at = datetime.now()
        
        self.storage_service.save_diary_entry(entry)
        self.refresh_entries()
        popup.dismiss()

    def delete_entry(self, entry, instance):
        """Удаляет запись"""
        def confirm_delete(*args):
            self.storage_service.delete_diary_entry(entry.id)
            self.refresh_entries()
            confirm_popup.dismiss()
        
        confirm_popup = Popup(
            title="Подтверждение",
            content=BoxLayout(orientation='vertical'),
            size_hint=(0.6, 0.4)
        )
        
        content = confirm_popup.content
        content.add_widget(Label(text=f"Удалить запись '{entry.title}'?"))
        
        buttons_layout = BoxLayout(size_hint_y=None, height=50)
        cancel_btn = Button(text="Отмена")
        cancel_btn.bind(on_press=confirm_popup.dismiss)
        buttons_layout.add_widget(cancel_btn)
        
        delete_btn = Button(text="Удалить", background_color=(1, 0, 0, 1))
        delete_btn.bind(on_press=confirm_delete)
        buttons_layout.add_widget(delete_btn)
        
        content.add_widget(buttons_layout)
        confirm_popup.open()

    def go_back(self, instance):
        """Возвращается к камере"""
        app = App.get_running_app()
        app.root.current = 'camera'

class SettingsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage_service = StorageService()
        self.create_ui()

    def create_ui(self):
        """Создает интерфейс настроек"""
        layout = BoxLayout(orientation='vertical')
        
        # Заголовок
        layout.add_widget(Label(text="Настройки", size_hint_y=None, height=50))
        
        # Скролл для настроек
        scroll = ScrollView()
        settings_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10)
        settings_layout.bind(minimum_height=settings_layout.setter('height'))
        
        # Быстрый режим
        quick_mode_layout = BoxLayout(size_hint_y=None, height=50)
        quick_mode_layout.add_widget(Label(text="Быстрый режим", size_hint_x=0.7))
        self.quick_mode_switch = ToggleButton(text="Выкл", size_hint_x=0.3)
        self.quick_mode_switch.bind(on_press=self.toggle_quick_mode)
        quick_mode_layout.add_widget(self.quick_mode_switch)
        settings_layout.add_widget(quick_mode_layout)
        
        # Автозакрытие
        auto_close_layout = BoxLayout(size_hint_y=None, height=50)
        auto_close_layout.add_widget(Label(text="Автозакрытие (мин)", size_hint_x=0.7))
        self.auto_close_input = TextInput(text="5", multiline=False, size_hint_x=0.3)
        auto_close_layout.add_widget(self.auto_close_input)
        settings_layout.add_widget(auto_close_layout)
        
        # Шаблоны текста
        templates_header = Label(text="Шаблоны текста (макс. 5)", size_hint_y=None, height=30)
        settings_layout.add_widget(templates_header)
        
        # Форма добавления шаблона
        template_form = BoxLayout(orientation='vertical', size_hint_y=None, height=120)
        
        self.template_name_input = TextInput(hint_text="Название шаблона", size_hint_y=None, height=40)
        template_form.add_widget(self.template_name_input)
        
        self.template_content_input = TextInput(hint_text="Текст шаблона", size_hint_y=None, height=40)
        template_form.add_widget(self.template_content_input)
        
        add_template_btn = Button(text="Добавить шаблон", size_hint_y=None, height=40)
        add_template_btn.bind(on_press=self.add_template)
        template_form.add_widget(add_template_btn)
        
        settings_layout.add_widget(template_form)
        
        # Список шаблонов
        self.templates_list = BoxLayout(orientation='vertical', size_hint_y=None)
        self.templates_list.bind(minimum_height=self.templates_list.setter('height'))
        settings_layout.add_widget(self.templates_list)
        
        # Кнопка обновления списка шаблонов
        refresh_templates_btn = Button(text="Обновить шаблоны", size_hint_y=None, height=40)
        refresh_templates_btn.bind(on_press=self.refresh_templates)
        settings_layout.add_widget(refresh_templates_btn)
        
        # Карта
        map_layout = BoxLayout(size_hint_y=None, height=50)
        map_layout.add_widget(Label(text="Карта региона", size_hint_x=0.7))
        map_btn = Button(text="Открыть", size_hint_x=0.3)
        map_btn.bind(on_press=self.open_map)
        map_layout.add_widget(map_btn)
        settings_layout.add_widget(map_layout)
        
        # Сброс счетчика
        reset_counter_layout = BoxLayout(size_hint_y=None, height=50)
        reset_counter_layout.add_widget(Label(text="Сбросить счетчик фото", size_hint_x=0.7))
        reset_btn = Button(text="Сбросить", size_hint_x=0.3)
        reset_btn.bind(on_press=self.reset_counter)
        reset_counter_layout.add_widget(reset_btn)
        settings_layout.add_widget(reset_counter_layout)
        
        scroll.add_widget(settings_layout)
        layout.add_widget(scroll)
        
        # Кнопка возврата
        back_btn = Button(
            text="Назад",
            size_hint_y=None,
            height=50
        )
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
        
        # Загружаем шаблоны
        self.refresh_templates()

    def toggle_quick_mode(self, instance):
        """Переключает быстрый режим"""
        if instance.state == 'down':
            instance.text = "Вкл"
        else:
            instance.text = "Выкл"

    def add_template(self, instance):
        """Добавляет шаблон текста"""
        name = self.template_name_input.text.strip()
        content = self.template_content_input.text.strip()
        
        if not name or not content:
            popup = Popup(
                title="Ошибка",
                content=Label(text="Заполните все поля"),
                size_hint=(0.6, 0.4)
            )
            popup.open()
            return
        
        template = TemplateText(name, content)
        self.storage_service.save_text_template(template)
        
        # Очищаем поля
        self.template_name_input.text = ""
        self.template_content_input.text = ""
        
        # Обновляем список
        self.refresh_templates()
        
        popup = Popup(
            title="Успех",
            content=Label(text="Шаблон добавлен"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def refresh_templates(self, instance=None):
        """Обновляет список шаблонов"""
        # Очищаем текущий список
        self.templates_list.clear_widgets()
        
        # Получаем шаблоны
        templates = self.storage_service.get_text_templates()
        
        if not templates:
            self.templates_list.add_widget(Label(text="Нет шаблонов"))
            return
        
        # Добавляем шаблоны
        for template in templates:
            template_widget = BoxLayout(size_hint_y=None, height=60)
            
            # Название и содержание
            info_layout = BoxLayout(orientation='vertical')
            info_layout.add_widget(Label(text=template.name, halign='left'))
            content_preview = template.content[:30] + "..." if len(template.content) > 30 else template.content
            info_layout.add_widget(Label(text=content_preview, halign='left', font_size='12sp'))
            template_widget.add_widget(info_layout)
            
            # Кнопка удаления
            delete_btn = Button(text="Удалить", size_hint_x=None, width=80, background_color=(1, 0, 0, 1))
            delete_btn.bind(on_press=partial(self.delete_template, template))
            template_widget.add_widget(delete_btn)
            
            self.templates_list.add_widget(template_widget)

    def delete_template(self, template, instance):
        """Удаляет шаблон"""
        self.storage_service.delete_text_template(template.id)
        self.refresh_templates()

    def open_map(self, instance):
        """Открывает карту"""
        app = App.get_running_app()
        app.root.current = 'map'

    def reset_counter(self, instance):
        """Сбрасывает счетчик фото"""
        # В реальном приложении это будет сохраняться в настройках
        popup = Popup(
            title="Успех",
            content=Label(text="Счетчик сброшен"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def go_back(self, instance):
        """Возвращается к камере"""
        app = App.get_running_app()
        app.root.current = 'camera'

class MapScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_ui()

    def create_ui(self):
        """Создает интерфейс карты"""
        layout = BoxLayout(orientation='vertical')
        
        # Заголовок
        layout.add_widget(Label(text="Карта GPS точек", size_hint_y=None, height=50))
        
        # Область карты (заглушка)
        map_area = BoxLayout(orientation='vertical')
        map_area.add_widget(Label(text="Карта GPS точек", font_size='20sp'))
        map_area.add_widget(Label(text="Здесь будут отображаться точки съемки", color=(0.5, 0.5, 0.5, 1)))
        layout.add_widget(map_area)
        
        # Кнопки управления
        controls_layout = BoxLayout(size_hint_y=None, height=50)
        
        my_location_btn = Button(text="Моя позиция")
        my_location_btn.bind(on_press=self.center_on_my_location)
        controls_layout.add_widget(my_location_btn)
        
        all_points_btn = Button(text="Все точки")
        all_points_btn.bind(on_press=self.show_all_points)
        controls_layout.add_widget(all_points_btn)
        
        layout.add_widget(controls_layout)
        
        # Кнопка возврата
        back_btn = Button(
            text="Назад",
            size_hint_y=None,
            height=50
        )
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)

    def center_on_my_location(self, instance):
        """Центрирует карту на текущей позиции"""
        popup = Popup(
            title="Информация",
            content=Label(text="Центрирование на текущей позиции"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def show_all_points(self, instance):
        """Показывает все точки"""
        popup = Popup(
            title="Информация",
            content=Label(text="Показ всех точек GPS"),
            size_hint=(0.6, 0.4)
        )
        popup.open()

    def go_back(self, instance):
        """Возвращается к настройкам"""
        app = App.get_running_app()
        app.root.current = 'settings'