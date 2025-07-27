import json
import os
import sys
import fitz  # PyMuPDF per leggere i PDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QWidget,
                             QVBoxLayout, QLineEdit, QTextEdit, QFileDialog, QProgressBar)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Importazioni LangChain con OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imposta l'API Key di OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Errore: L'API Key di OpenAI non è impostata. Assicurati di averla definita nella variabile d'ambiente.")

class PdfLoaderThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            text = ""
            doc = fitz.open(self.file_path)
            num_pages = len(doc)

            for i, page in enumerate(doc):
                text += page.get_text()
                self.progress.emit(int(((i + 1) / num_pages) * 100))
                self.msleep(50)

            print("LUNGHEZZA TESTO PDF: ",len(text))
            self.finished.emit(text.strip() if text.strip() else "Errore: Il PDF non contiene testo leggibile.")
        except Exception as e:
            self.finished.emit(f"Errore nel caricamento del PDF: {e}")

class IndexingThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, pdf_text):
        super().__init__()
        self.pdf_text = pdf_text

    def run(self):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            documents = splitter.create_documents([self.pdf_text])
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)

            vectorstore.save_local("faiss_index_OpenAI")

            self.finished.emit(vectorstore)

        except Exception as e:
            print(f"Errore durante l'indicizzazione: {e}")
            self.finished.emit(None)


class QueryThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, user_input, qa_chain, chat_model):
        super().__init__()
        self.user_input = user_input
        self.qa_chain = qa_chain
        self.chat_model = chat_model

    def run(self):
        try:
            if not self.qa_chain:
                self.finished.emit("Carica prima un PDF.")
                return

            relevant_docs = self.qa_chain.invoke(self.user_input)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            print("CONTEXT LEN: ", len(context))
            print("CONTEXT: ", context)

            messages = [
                {"role": "system",
                 "content": "Sei un assistente AI che risponde a domande basandoti sul contenuto di un documento."
                            " Rispondi in modo dettagliato e preciso, approfondendo il contenuto del testo"},
                {"role": "system", "content": f"Il documento fornisce queste informazioni rilevanti:\n{context}"},
                {"role": "user", "content": f"Domanda: {self.user_input}"}
            ]

            response = self.chat_model.invoke(messages)

            if hasattr(response, 'content'):
                self.finished.emit(response.content)
            else:
                self.finished.emit(str(response))

        except Exception as e:
            self.finished.emit(f"Errore durante la generazione della risposta: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatBot OpenAI GPT-3.5")
        self.setGeometry(250, 250, 1920, 1080)
        self.setWindowIcon(QIcon("ChatBotIcon.PNG"))
        self.setStyleSheet("background-color: #aed8f5")
        self.pdf_text = ""
        self.qa_chain = None
        self.chat_model = ChatOpenAI(model="gpt-3.5-turbo")
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignTop)

        self.label = QLabel("Ciao! Sono il tuo ChatBot e sono qui per aiutarti", self)
        self.label.setFont(QFont("Arial", 15))
        self.label.setStyleSheet("background-color: #d9f2d3; border-radius: 35px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(980, 80)
        self.main_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        # Etichetta per il titolo del PDF
        self.pdf_title_label = QLabel("", self)  # Inizialmente vuota
        self.pdf_title_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.pdf_title_label.setAlignment(Qt.AlignCenter)
        self.pdf_title_label.setStyleSheet("color: #333; background-color: #fff8dc; padding: 5px; border-radius: 10px;")
        self.pdf_title_label.setFixedSize(880, 40)  # Dimensioni fisse per il titolo
        self.pdf_title_label.setVisible(False)  # Non visibile finché non viene caricato un PDF
        self.main_layout.addWidget(self.pdf_title_label, alignment=Qt.AlignCenter)

        # "Carica PDF" button
        self.load_pdf_button = QPushButton("Carica PDF", self)
        self.load_pdf_button.setFont(QFont("Arial", 12))
        self.load_pdf_button.setFixedSize(200, 80)

        self.load_pdf_button.setStyleSheet(
            "QPushButton:hover { background-color: #c2e0f4; }"
        )
        self.load_pdf_button.clicked.connect(self.load_pdf)
        self.main_layout.addWidget(self.load_pdf_button, alignment=Qt.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Scrivi qui la tua domanda...")
        self.input_box.setFont(QFont("Arial", 14))
        self.input_box.setStyleSheet("padding: 10px; border-radius: 10px;")
        self.main_layout.addWidget(self.input_box)

        self.output_box = QTextEdit(self)
        self.output_box.setFont(QFont("Arial", 12))
        self.output_box.setStyleSheet("padding: 10px; background-color: #f0f8ff; border-radius: 10px;")
        self.output_box.setReadOnly(True)
        self.main_layout.addWidget(self.output_box)

        self.query_button = QPushButton("Invia", self)
        self.query_button.setFont(QFont("Arial", 12))
        self.query_button.setFixedSize(600, 80)

        self.query_button.setStyleSheet(
            "QPushButton:hover {background-color: #c2e0f4; }"
        )
        self.query_button.clicked.connect(self.ask_chatbot)
        self.main_layout.addWidget(self.query_button, alignment=Qt.AlignCenter)

        self.label.setText("Ciao, sono il tuo ChatBot basato sul modello GPT-3.5-Turbo di OpenAI")
        self.load_existing_index()

    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Carica PDF", "C:/Users/simo-/OneDrive/Desktop", "PDF Files (*.pdf)")
        if file_path:
            filename = os.path.basename(file_path)
            self.save_filename(filename)
            self.label.setText("Attendi mentre carico il pdf e lo elaboro...")
            self.pdf_title_label.setText(filename)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.pdf_loader_thread = PdfLoaderThread(file_path)
            self.pdf_loader_thread.progress.connect(self.progress_bar.setValue)
            self.pdf_loader_thread.finished.connect(self.on_pdf_loaded)
            self.pdf_loader_thread.start()

    def save_filename(self, filename):
        with open("filename_1.json", "w") as f:
            json.dump({"filename": filename}, f)

    def on_pdf_loaded(self, text):
        self.progress_bar.setVisible(False)
        self.pdf_text = text
        if "Errore" in text:
            self.label.setText(text)
        else:
            self.label.setText("PDF caricato! Attendi prima che elabori il documento...")

            with open("filename_1.json", "r") as f:
                data = json.load(f)
                filename = data.get("filename", "Nome file non trovato")
                self.pdf_title_label.setText(filename)
                self.pdf_title_label.setVisible(True)

            self.indexing_thread = IndexingThread(text)
            self.indexing_thread.finished.connect(self.on_indexing_finished)
            self.indexing_thread.start()

    def on_indexing_finished(self, vectorstore):
        if vectorstore:
            self.label.setText("Indice creato! Ora puoi fare domande sul documento.")
            self.qa_chain = vectorstore.as_retriever(search_kwargs={"k": 20})
        else:
            self.label.setText("Errore nella creazione dell'indice.")

    def load_existing_index(self):
        if os.path.exists("faiss_index_OpenAI"):
            vectorstore = FAISS.load_local("faiss_index_OpenAI", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            self.qa_chain = vectorstore.as_retriever(search_kwargs={"k": 20})
            print("Indice FAISS caricato correttamente.")
        try:
         with open("filename_1.json", "r") as f:
            data = json.load(f)
            filename = data.get("filename", "Nome file non trovato")
            self.pdf_title_label.setText(filename)
            self.pdf_title_label.setVisible(True)
        except Exception as e:
            print(f"Error: {e}")

    def ask_chatbot(self):
        user_input = self.input_box.text().strip()

        if not user_input:
            self.output_box.setText("Per favore, scrivi una domanda.")
            return

        self.output_box.setText("Sto elaborando la risposta...")
        QApplication.processEvents()

        if self.qa_chain is None:
            self.load_existing_index()

        if self.qa_chain is None:
            self.output_box.setText("Non ho nessun file caricato. Per favore, carica un PDF.")
            return

        self.query_thread = QueryThread(user_input, self.qa_chain, self.chat_model)
        self.query_thread.finished.connect(self.output_box.setText)
        self.query_thread.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
