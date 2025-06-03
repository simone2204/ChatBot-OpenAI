import json
import os
import sys
import fitz  # Per leggere i PDF
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QWidget,
                             QVBoxLayout, QLineEdit, QTextEdit, QFileDialog, QProgressBar)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from huggingface_hub import InferenceClient
from langchain_community.embeddings import OpenAIEmbeddings

# Importazioni LangChain con Hugging Face
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Imposta la tua API Key di Hugging Face
HF_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_API_KEY:
    raise ValueError(
        "Errore: L'API Key di Hugging Face non è impostata. Assicurati di averla definita nella variabile d'ambiente.")


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

            if num_pages == 0:
                self.finished.emit("Errore: Il PDF è vuoto o non può essere letto.")
                return

            for i, page in enumerate(doc):
                text += page.get_text()
                progress_value = int(((i + 1) / num_pages) * 100)
                self.progress.emit(progress_value)
                self.msleep(50)

            if not text.strip():
                self.finished.emit("Errore: Il PDF non contiene testo leggibile.")
            else:
                self.finished.emit(text.strip())

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

            vectorstore.save_local("faiss_index_LLAMA")

            self.finished.emit(vectorstore)

        except Exception as e:
            print(f"Errore durante l'indicizzazione: {e}")
            self.finished.emit(None)


class QueryThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, qa_chain, inference_client, user_input):
        super().__init__()
        self.qa_chain = qa_chain
        self.inference_client = inference_client
        self.user_input = user_input

    def run(self):
        try:
            relevant_docs = self.qa_chain.invoke(self.user_input)

            if not relevant_docs:
                self.finished.emit("La domanda non è pertinente.")
                return

            context = "\n".join([doc.page_content for doc in relevant_docs])
            print("CONTEXT: ", len(context))
            print(context)

            prompt = f"""
            <s>Source: system

            Sei un assistente AI che fornisce risposte ad un utente basandoti solo ed esclusivamente sulle informazioni nel CONTESTO.
            L'utente farà una domanda, e tu dovrai rispondere in maniera accurata ed approfondita in base a tutte le informazioni pertinenti che riuscirai a trovare nel CONTESTO.  
            NON inventare nulla. Se non trovi informazioni rilevanti nel CONTESTO, rispondi semplicemente con:  
            "Non ho trovato informazioni rilevanti nel documento."
            Se l'utente ti chiede un parere personale su un argomento, rispondi giudicando l'argomento in base alle tue conoscenze e ai tuoi gusti personali, 
            facendo riferimento, ove possibile, alle informazioni che trovi nel CONTESTO. Se non trovi nessuna informazione relativa nel CONTESTO ad una domanda personale dell'utente, rispondi
            giudicando l'argomento in base alle tue conoscenze personali.

            <step> Source: user

            ### Contesto del documento:
            {context}

            <step> Source: assistant

            <s>Source: user

            ### Domanda dell'utente:
            {self.user_input}

            <step> Source: assistant
            Destination: user
            """

            response = self.inference_client.text_generation(prompt)

            generated_text = response if isinstance(response, str) else "Errore nella risposta del modello."
            self.finished.emit(generated_text)

        except Exception as e:
            self.finished.emit(f"Errore durante l'elaborazione: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatBot LLAMA - Hugging Face")
        self.setGeometry(250, 250, 1920, 1080)
        self.setWindowIcon(QIcon("ChatBotIcon.PNG"))
        self.setStyleSheet("background-color: #aed8f5")
        self.pdf_text = ""
        self.qa_chain = None
        self.faiss_path = "faiss_index_LLAMA"
        self.inference_client = InferenceClient(
            model="meta-llama/Llama-3.2-3B-Instruct",
            # meta-llama/Llama-3.2-3B-Instruct  # meta-llama/Llama-3.1-8B-Instruct   # meta-llama/Llama-3.3-70B-Instruct
            token=HF_API_KEY
        )
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
        self.label.setFixedSize(880, 80)
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
            "background-color: #e3f3fd; border-radius: 15px; padding: 10px;"
            "cursor: pointer;"
            "transition: background-color 0.3s ease;"
        )
        self.load_pdf_button.setStyleSheet(
            "QPushButton:hover { background-color: #c2e0f4; }"
        )
        self.load_pdf_button.clicked.connect(self.load_pdf)
        self.main_layout.addWidget(self.load_pdf_button, alignment=Qt.AlignCenter)

        # Progress BAR del pdf
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
        self.query_button.setStyleSheet("background-color: #e3f3fd; border-radius: 15px; padding: 100px;"
                                        "cursor: pointer;"
                                        "transition: background-color 0.3s ease;")
        self.query_button.setStyleSheet(
            "QPushButton:hover {background-color: #c2e0f4; }"
        )
        self.query_button.clicked.connect(self.ask_chatbot)
        self.main_layout.addWidget(self.query_button, alignment=Qt.AlignCenter)

        self.label.setText("Ciao, sono il tuo ChatBot basato sul modello LLAMA di Meta AI")
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
        with open('filename_2.json', 'w') as f:
            json.dump({"filename": filename}, f)

    def on_pdf_loaded(self, text):
        self.progress_bar.setVisible(False)
        self.pdf_text = text
        if "Errore" in text:
            self.label.setText(text)
        else:
            self.label.setText("PDF caricato! Attendi prima che elabori il documento...")

            with open("filename_2.json", "r") as f:
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
            self.qa_chain = vectorstore.as_retriever(search_kwargs={"k": 10})
        else:
            self.label.setText("Errore nella creazione dell'indice.")

    def load_existing_index(self):
        if os.path.exists("faiss_index_LLAMA"):
            vectorstore = FAISS.load_local("faiss_index_LLAMA", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            self.qa_chain = vectorstore.as_retriever(search_kwargs={"k": 10})
            print("Indice FAISS caricato correttamente.")

        try:
         with open("filename_2.json", "r") as f:
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

        self.query_thread = QueryThread(self.qa_chain, self.inference_client, user_input)
        self.query_thread.finished.connect(self.output_box.setText)
        self.query_thread.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()