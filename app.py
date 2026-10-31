# importng dependencies

from kivy.config import Config

# Disabling the multitouch "red dots" and force use of standard mouse
Config.set('input', 'mouse', 'mouse,disable_multitouch')
# Stoping Kivy from looking at raw system touch events that cause hovering issues
Config.set('input', '%(name)s', ' ')


from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.factory import Factory
from datetime import datetime
from kivy.clock import Clock

import cv2
import csv
import numpy as np
import tflite_runtime.interpreter as tflite

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


class HomeScreen(Screen):

    # Loading the database and the model
    def __init__(self, **kw):
        super().__init__(**kw)
        
        self.database = np.load('./db/database_embeddings.npz')
        self.known_matricules = self.database.files

        self.interpreter = tflite.Interpreter(model_path="./db/verifyme.tflite")
        self.interpreter.allocate_tensors()

    # Verify if student information
    def verify_student_data(self, matricule):

        total_required = 50000
        is_paid = False
        student_name = "Unknown"
        
        try:
            with open('./db/Fee.csv', mode='r') as f: 
                reader = csv.DictReader(f)
                for row in reader:
                    if row['MATRICULE'] == matricule:
                        amount_paid = int(row['FEES PAID'])
                        student_name = row['NAME']
                        is_paid = (amount_paid >= total_required)
                        break
        except FileNotFoundError:
            return "error_file_missing", "Unknown"
        
        if not is_paid:
            return "fees_incomplete", student_name

        # Verifying Enrollment
        app = App.get_running_app()
        course = app.current_course 
        is_enrolled = False
        
        try:
            with open('./db/Enrolled.csv', mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get(course) == matricule:
                        is_enrolled = True
                        break
        except Exception as e:
            print(f"Enrollment Check Error: {e}")
        
        if not is_enrolled:
            return "not_enrolled", student_name

        return "success", student_name

    # Capturing images from webcam and performing verifications
    def capture(self):
        camera = self.ids['camera']

        # Mapping the display labels
        name_disp = self.ids['display_name']
        matric_disp = self.ids['display_matricule']
        fee_disp = self.ids['fee_status']
        course_disp = self.ids['course_status']

        if not camera.texture:
            return

        # Image Conversion & Preprocessing
        texture = camera.texture
        w, h = texture.size
        frame = np.frombuffer(texture.pixels, dtype=np.uint8).reshape((h, w, 4))
        img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 0)

        # Crop to the face guide area
        side = min(w, h)
        start_x, start_y = (w - side) // 2, (h - side) // 2
        cropped = img[start_y:start_y+side, start_x:start_x+side]

        # TRIGGER: <coveredCamera@Popup>
        avg_brightness = np.mean(cropped)
        if avg_brightness < 50:
            Factory.coveredCamera().open()
            return

        # Model Inferencing
        processed_img = cv2.resize(cropped, (100, 100))
        processed_img = processed_img.astype(np.float32) / 255.0
        input_data = np.expand_dims(processed_img, axis=0)

        input_index = self.interpreter.get_input_details()[0]['index']
        output_index = self.interpreter.get_output_details()[0]['index']
        self.interpreter.set_tensor(input_index, input_data)
        self.interpreter.invoke()
        live_embedding = self.interpreter.get_tensor(output_index).flatten()

        # Matching Logic
        best_match = None
        max_similarity = 0.0
        threshold = 0.9

        for matricule in self.database.files:
            stored_embedding = self.database[matricule].flatten()
            l1_dist = np.sum(np.abs(live_embedding - stored_embedding))
            similarity = np.exp(-l1_dist / 400) 
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = matricule

        # TRIGGER: <unidentified@Popup>
        if not best_match or max_similarity < threshold:
            Factory.unidentified().open()
            return

        # Verifying both fees and enrollment
        status, student_name = self.verify_student_data(best_match)

        # TRIGGER: <incompleteFees@Popup>
        if status == "fees_incomplete":
            name_disp.text = f"Name: {student_name}"
            matric_disp.text = f"Matricule: {best_match}"
            fee_disp.text = f"Fee Status: Incomplete"
            course_disp.text = f"Course Status: Registered"
            Factory.incompleteFees().open()
            return
            
        # TRIGGER: <notRegistered@Popup>
        elif status == "not_enrolled":
            Factory.notRegistered().open()
            return

        # Verifying if student is already in the list
        app = App.get_running_app()
        dashboard = app.root.get_screen('dashboard')
        if any(record[1] == best_match for record in dashboard.attendance_data):
            Factory.alreadyVerified().open()
            return

        # Logging Attendance
        timestamp = datetime.now().strftime('%H:%M:%S')
        name_disp.text = f"Name: {student_name}"
        matric_disp.text = f"Matricule: {best_match}"
        fee_disp.text = f"Fee Status: Paid"
        course_disp.text = f"Course Status: Registered"

        # Switching context to the dashboard screen to add the row
        dashboard.add_student(student_name, best_match, timestamp)
        


class Dashboard(Screen):
    
    # Collecting attendance data for PDF
    attendance_data = []

    def add_student(self, name, matricule, time_stamp):
        # Store data for PDF
        self.attendance_data.append([name, matricule, time_stamp])

        name_label = Label(
            text=name,
            size_hint_x=0.5,
            size_hint_y=None,
            height='40dp',
            halign='left',
            valign='middle'
        )
        name_label.bind(size=lambda instance, value: setattr(instance, 'text_size', value))

        matricule_label = Label(
            text=matricule,
            size_hint_x=0.25,
            size_hint_y=None,
            height='40dp',
            halign='center',
            valign='middle'
        )
        matricule_label.bind(size=lambda instance, value: setattr(instance, 'text_size', value))

        time_label = Label(
            text=time_stamp,
            size_hint_x=0.25,
            size_hint_y=None,
            height='40dp',
            halign='right',
            valign='middle',
            padding= [0, 0, 15, 0]
        )
        time_label.bind(size=lambda instance, value: setattr(instance, 'text_size', value))

        # Displaying student data
        self.ids.attendance_list.add_widget(name_label)
        self.ids.attendance_list.add_widget(matricule_label)
        self.ids.attendance_list.add_widget(time_label)

    # Clear attendance list
    def clear_attendance(self):

        # Verifying if list is empty before clearing
        if not self.attendance_data:

            Factory.emptyList().open()
            return

        self.attendance_data = []
        self.ids.attendance_list.clear_widgets()

        Factory.clearedAttendance().open()

    # Generating PDF report
    def generate_pdf_report(self):
        
        # Verifying if list is empty before exporting
        if not self.attendance_data:

            Factory.emptyList().open()
            return

        app = App.get_running_app()
        course_code = app.current_course
        course_name = "Unknown Course"
        instructor = "Unknown Instructor"

        # Fetching course details from Courses.csv
        try:
            with open('./db/Courses.csv', mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Course Code'].strip().upper() == course_code:
                        course_name = row['Course Title'].upper()
                        instructor = row.get('Instructor', row.get('Course Instructor(s)', "Unknown")).upper()
                        break
        except Exception as e:
            print(f"Error fetching course details: {e}")

        filename = f"{course_code} Attendance List.pdf"
        
        doc = SimpleDocTemplate(filename, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        elements = []

        # Attendance report title
        styles = getSampleStyleSheet()
        
        # Updated font size to 14 as requested previously and maintained leading for spacing
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=12,
            leading=22 
        )

        # Creating the header
        logo_path = "./db/logo.png"
        try:
            logo = Image(logo_path, width=80, height=80)
        except:
            logo = Paragraph("LOGO MISSING", styles['Normal'])

        header_left = Paragraph(
            f"<b>COURSE:</b> <br/>{course_name}<br/><b>COURSE CODE:</b> <br/>{course_code}", 
            header_style
        )
        header_right = Paragraph(
            f"<b>INSTRUCTOR:</b> <br/>{instructor}<br/><b>DATE:</b> <br/>{datetime.now().strftime('%d/%m/%Y')}", 
            header_style
        )

        header_table = Table(
            [[header_left, logo, header_right]], 
            colWidths=[230, 80, 230]
        )
        
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (0,0), 'LEFT'), 
            ('ALIGN', (1,0), (1,0), 'CENTER'),
            ('ALIGN', (2,0), (2,0), 'RIGHT'),  
            ('LINEABOVE', (0,0), (-1,0), 1, colors.black),
            ('LINEBELOW', (0,0), (-1,0), 1, colors.black), 
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 24))

        # Dynamic Heading
        elements.append(Paragraph(f"<b>ATTENDANCE LIST FOR {course_code}</b>", styles['Title']))
        elements.append(Spacer(1, 15))

        # Preparing table data
        table_data = [["S/N", "NAME", "MATRICULE", "TIME"]]
        for index, record in enumerate(self.attendance_data, start=1):
            table_data.append([str(index), record[0], record[1], record[2]])

        report_table = Table(table_data, colWidths=[40, 260, 140, 100])

        # Styling the attendance table
        style = TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 12),  # Set table font to 12
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ])
        report_table.setStyle(style)
        elements.append(report_table)

        # Generating the pdf
        try:
            doc.build(elements)
            Factory.savedSuccessfully().open()

        except Exception as e:
            print(f"PDF Error: {e}")
            Factory.errorSaving().open()


class WindowManager(ScreenManager):
    pass

# Loading different pages or files
kv_file = Builder.load_file('./screens.kv')

class VerifyMeApp(App):

    # Current Course Code
    current_course = None

    def build(self):
        
        # Scheduling Popup Window to display after 5 seconds
        Clock.schedule_once(self.show_course_popup, 5)
        return kv_file

    def show_course_popup(self, dt):
        # TRIGGER: <courseCode@Popup>
        Factory.courseCode().open()

    def verify_course_code(self, code, popup_instance):
        code = code.strip().upper()
        valid = False
        
        # Verifying if the course code is valid
        try:
            with open('./db/Courses.csv', mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Course Code'].strip().upper() == code:
                        valid = True
                        break
        except FileNotFoundError:
            print("Error: ./db/Courses.csv not found!")
            
        if valid:
            self.current_course = code
            popup_instance.dismiss()
        else:
            # TRIGGER: <invalidCourse@Popup>
            Factory.invalidCourse().open()
    
if __name__ == '__main__':
    VerifyMeApp().run()
