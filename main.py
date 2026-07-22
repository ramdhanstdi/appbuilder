import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

# 1. Definisikan Tool (Tangan untuk AI)
@tool
def write_code_file(file_path: str, content: str) -> str:
    """
    PENTING: Panggil fungsi ini HANYA ketika kamu perlu membuat atau menyimpan file kode.
    Fungsi ini akan otomatis membuat folder jika path-nya belum ada.
    """
    try:
        # Ekstrak nama direktori dari path yang diberikan AI
        directory = os.path.dirname(file_path)
        
        # Buat struktur folder jika belum ada
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Tulis kode ke dalam file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"SUKSES: File berhasil disimpan di {file_path}"
    
    except Exception as e:
        return f"GAGAL: Terjadi error sistem saat menulis file - {str(e)}"
llm = ChatOpenAI(
    openai_api_base="http://localhost:20128/v1", 
    openai_api_key="sk-199156432860867f-1xcg14-e4f1dc4e", 
    model_name="cc/claude-sonnet-5(high)",
    temperature=0.1
)
llm_with_tools = llm.bind_tools([write_code_file])

# ==========================================
# BAGIAN 4: Definisikan State & Memori
# ==========================================
# State ini berfungsi sebagai "memori" percakapan.
# add_messages memastikan pesan baru selalu ditambahkan ke memori, bukan menimpa yang lama.
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ==========================================
# BAGIAN 5: Definisikan Nodes (Titik Proses)
# ==========================================
# Node 1: Project Manager (AI) - versi aman dari crash
def project_manager(state: State):
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        print(f"\n⚠️  [System Warning]: Model menolak request atau bermasalah: {str(e)}")
        # Mengembalikan pesan fallback agar siklus chat tidak terputus
        from langchain_core.messages import AIMessage
        fallback_msg = AIMessage(content="Maaf, model sedang memblokir instruksi atau terjadi masalah koneksi ke provider 9router. Coba sederhanakan permintaanmu atau ganti model di 9router.")
        return {"messages": [fallback_msg]}

# Node 2: Eksekutor Tool (File Writer)
# LangGraph punya fungsi bawaan untuk mengeksekusi tool yang dipanggil oleh AI
tools_node = ToolNode([write_code_file])

# Fungsi Routing (Logika Pengambilan Keputusan)
def route_tools(state: State):
    last_message = state["messages"][-1]
    # Jika PM memutuskan perlu membuat file kode, arahkan ke Node "tools"
    if last_message.tool_calls:
        return "tools"
    # Jika tidak ada pemanggilan tool, berarti PM ingin bertanya/menjawab ke User
    return END

# ==========================================
# BAGIAN 6: Merakit Graph & Terminal Loop (Human-in-the-Loop)
# ==========================================
workflow = StateGraph(State)

# 1. Daftarkan Nodes ke Graph
workflow.add_node("pm", project_manager)
workflow.add_node("tools", tools_node)

# 2. Tentukan Alur Kerja (Edges)
workflow.add_edge(START, "pm") # Mulai dari PM
workflow.add_conditional_edges("pm", route_tools, {"tools": "tools", END: END})
workflow.add_edge("tools", "pm") # Setelah file dibuat, kembalikan laporannya ke PM

# 3. Compile (Build) sistemnya
app = workflow.compile()

# ==========================================
# BAGIAN 7: Jalankan di Terminal
# ==========================================
print("🚀 App Builder Agent (PM) Siap! Ketik 'quit' untuk keluar.")
print("Cobalah: 'Tolong buatkan komponen React Button sederhana di folder ./my_app/src/components/'\n")

while True:
    user_input = input("👨‍💻 Kamu: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
        
    # Memasukkan input user ke dalam memori graf
    for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
        for node_name, value in event.items():
            # Jika yang merespon adalah PM, tampilkan teksnya
            if node_name == "pm":
                ai_message = value["messages"][-1].content
                if ai_message: # Hanya print jika ada teks balasan (bukan trigger tool)
                    print(f"🤖 PM: {ai_message}")
            
            # Jika sistem sedang mengeksekusi pembuatan file
            elif node_name == "tools":
                print(f"⚙️  Sistem: Sedang membuat struktur file...")