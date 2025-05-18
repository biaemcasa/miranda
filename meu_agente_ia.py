import os
from dotenv import load_dotenv
from google import genai
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types  # Para criar conteúdos (Content e Part)
from datetime import date
import textwrap  # Para formatar melhor a saída de texto
import requests  # Para fazer requisições HTTP
import warnings

warnings.filterwarnings("ignore")

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configura a API Key do Google Gemini
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    print("A chave da API do Google Gemini não foi encontrada. Certifique-se de ter um arquivo .env com GOOGLE_API_KEY definido.")
    exit()

# Configura o cliente da SDK do Gemini
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
client = genai.Client()
MODEL_ID = "gemini-2.0-flash"

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
            for part in event.content.parts:
                if part.text is not None:
                    final_response += part.text
                    final_response += "\n"
    return final_response

# Função auxiliar para exibir texto formatado
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

##########################################
# --- Agente 1: Buscador de Notícias --- #
##########################################
def agente_buscador(topico, data_de_hoje, links_produtos=None):
    buscador = Agent(
        name="agente_buscador",
        model="gemini-2.0-flash",
        description="Agente que busca informações no Google",
        tools=[google_search],
        instruction="""
        Você é um assistente de pesquisa. A sua tarefa é usar a ferramenta do google (google_search)
        para recueprar as últimas notícias de lançamentos muito relevantes sobre o tópico abaixo.
        Se forem fornecidos links de produtos, use esses links como ponto de partida para entender
        os produtos específicos do lojista e buscar notícias e informações relevantes sobre eles ou a categoria a que pertencem.
        Foque em no máximo 5 lançamentos relevantes...
        """
    )
    entrada_do_agente_buscador = f"Tópico: {topico}\nData de hoje: {data_de_hoje}"
    if links_produtos:
        entrada_do_agente_buscador += f"\nLinks de produtos de referência: {', '.join(links_produtos)}"

    lancamentos = call_agent(buscador, entrada_do_agente_buscador)
    return lancamentos

################################################
# --- Agente 2: Planejador de posts --- #
################################################
def agente_planejador(topico, lancamentos_buscados):
    planejador = Agent(
        name="agente_planejador",
        model="gemini-2.0-flash",
        # Inserir as instruções do Agente Planejador #################################################
        instruction="""
        Você é um planejador de conteúdos, especialista em criar posts para blogs de ecommerce.
        Com base na lista de lançamentos mais recentes e relevantes buscados, você deve:
        usar a ferramenta de busca do Google (google_search) para criar um plano sobre
        quais são os pontos mais relevantes que poderíamos abordar em um texto de blog sobre cada um deles.
        Você também pode usar o (google_search) para encontrar mais informações sobre os temas e aprofundar.
        Ao final, você irá escolher o tema mais relevante entre eles com base nas suas pesquisas
        e retornar esse tema, seus pontos mais relevantes, e um plano com os assuntos a serem
        abordados no texto para blog que será escrito posteriormente.
        """,
        description="Agente que planeja posts",
        tools=[google_search]
    )

    entrada_do_agente_planejador = f"Tópico:{topico}\nLançamentos buscados: {lancamentos_buscados}"
    # Executa o agente
    plano_do_post = call_agent(planejador, entrada_do_agente_planejador)
    return plano_do_post

######################################
# --- Agente 3: Redator do Post --- #
######################################
def agente_redator(topico, plano_de_post):
    redator = Agent(
        name="agente_redator",
        model="gemini-2.0-flash",
        instruction="""
            Você é um Redator Criativo especializado em criar posts virais para blogs de moda.
            Você escreve posts para blogs de moda ligados a ecommerces. Utilize o tema fornecido no plano de post e os pontos mais relevantes
            fornecidos e, com base nisso, escreva um rascunho de post para blog sobre o tema indicado.
            O tamanho ideal para um texto de blog otimizado geralmente gira em torno de 1.500 a 2.000 palavras.
            O post deve ser engajador, informativo, com linguagem simples e incluir 2 a 4 tags no final.
            """,
        description="Agente redator de posts engajadores para blogs de moda"
    )
    entrada_do_agente_redator = f"Tópico: {topico}\nPlano de post: {plano_de_post}"
    # Executa o agente
    rascunho = call_agent(redator, entrada_do_agente_redator)
    return rascunho

############################################
# --- Agente 5: Buscador de Imagens --- #
############################################
def agente_buscador_imagens(topico, rascunho_gerado):
    buscador_imagens = Agent(
        name="agente_buscador_imagens",
        model="gemini-2.0-flash",
        instruction="""
            Você é um buscador de imagens para posts de blog de moda.
            Sua tarefa é usar a ferramenta de busca do Google (google_search) para encontrar imagens de alta qualidade
            e relevantes para ilustrar um post de blog sobre o tópico e rascunho fornecidos.
            Foque em encontrar imagens de sites e blogs de moda confiáveis.
            Para cada imagem encontrada, forneça a URL da imagem e a URL da página onde ela foi encontrada.
            Liste no máximo 5 imagens relevantes.
            """,
        description="Agente que busca imagens para posts de blog de moda",
        tools=[google_search]  # Use a ferramenta de busca do Google
    )
    entrada_do_agente_buscador_imagens = f"Tópico: {topico}\nRascunho do post: {rascunho_gerado}"
    # Executa o agente
    imagens_encontradas = call_agent(buscador_imagens, entrada_do_agente_buscador_imagens)
    return imagens_encontradas

##########################################
# --- Agente 4: Revisor de Qualidade --- #
##########################################
def agente_revisor(topico, rascunho_gerado):
    revisor = Agent(
        name="agente_revisor",
        model="gemini-2.0-flash",
        instruction="""
            Você é um Editor e Revisor de Conteúdo meticuloso, especializado em posts virais para blogs de moda, orientados ao SEO do Google, com foco em blogs de moda.
            Por ter um público jovem, entre 21 a 35 anos, use um tom de escrita adequado.
            Revise o rascunho de post blogs de moda, orientados ao SEO do Google abaixo sobre o tópico indicado, verificando clareza, concisão, correção e tom.
            Além disso, avalie o post com foco em SEO para o Google, considerando os seguintes aspectos:
            - **Uso de Palavras-chave:** O rascunho utiliza palavras-chave relevantes para o tópico de forma natural no título, subtítulos e ao longo do texto? Sugira palavras-chave adicionais relevantes se necessário.
            - **Estrutura do Conteúdo:** O texto está bem estruturado com títulos (H1, H2, H3) e parágrafos curtos para facilitar a leitura?
            - **Links Internos e Externos:** O rascunho inclui oportunidades para links internos (para outros produtos ou posts do blog do ecommerce) e links externos (para fontes relevantes e confiáveis)? Sugira onde links podem ser adicionados.
            - **Legibilidade:** O texto é fácil de ler para o público-alvo (linguagem simples, frases curtas)?
            - **Otimização de Imagens (Alt Text):** Se as imagens fossem incluídas, o rascunho fornece contexto suficiente para gerar bons "alt text" (descrição da imagem) com palavras-chave relevantes? (O Agente 6 cuidará da inclusão, mas o Agente 4 pode avaliar a base no rascunho).
            - **Intenção de Busca:** O conteúdo do post alinha-se com a provável intenção de busca do usuário ao procurar sobre o tópico?

            Se o rascunho estiver ótimo e otimizado para SEO, responda apenas 'O rascunho está ótimo, otimizado para SEO e pronto para publicar!'.
            Caso haja problemas de qualidade ou otimização para SEO, aponte-os de forma clara e sugira melhorias específicas para tornar o post mais forte para SEO.
            """,
        description="Agente revisor de post para blogs de moda, orientados ao SEO do Google."
    )
    entrada_do_agente_revisor = f"Tópico: {topico}\nRascunho: {rascunho_gerado}"
    # Executa o agente
    texto_revisado = call_agent(revisor, entrada_do_agente_revisor)
    return texto_revisado

######################################################
# --- Agente 6: Formatador de Post com Imagens --- #
######################################################
def agente_formatador_imagens(rascunho_gerado, imagens_encontradas):
    formatador = Agent(
        name="agente_formatador_imagens",
        model="gemini-2.0-flash",
        instruction="""
            Você é um formatador de posts de blog, especializado em moda.
            Sua tarefa é pegar o rascunho do post e a lista de imagens encontradas e gerar o texto final do post,
            incluindo o código de incorporação das imagens nos locais apropriados dentro do texto.

            Formato de saída: Markdown. Use o seguinte formato para incorporar imagens:
            ![Descrição da Imagem](URL_DA_IMAGEM)

            Analise o texto do rascunho para identificar seções onde uma imagem seria relevante para ilustrar o conteúdo.
            Para cada imagem na lista fornecida, tente encontrar um ponto relevante no texto para inseri-la.
            Use o tópico da imagem ou o contexto do texto para gerar uma breve "Descrição da Imagem" (alt text).
            Não insira todas as imagens se elas não parecerem relevantes para o texto. Foque naquelas que melhor complementam o conteúdo.
            Posicione as imagens de forma que quebrem o texto de forma natural, talvez após um parágrafo relevante.
            Inclua as tags originais do rascunho no final do post formatado.
            """,
        description="Agente que formata o post do blog, incluindo código de incorporação de imagens em Markdown."
    )
    entrada_do_agente_formatador = f"Rascunho do Post:\n{rascunho_gerado}\n\nImagens Encontradas:\n{imagens_encontradas}"
    # Executa o agente
    post_formatado = call_agent(formatador, entrada_do_agente_formatador)
    return post_formatado

if __name__ == "__main__":
    data_de_hoje = date.today().strftime("%d/%m/%Y")

    print(" Iniciando o Sistema de Criação de Posts para blogs de moda, orientados ao SEO do Google ")
    print()

    # --- Obter o Tópico do Usuário ---
    topico = input("❓ Por favor, digite o TÓPICO sobre o qual você quer criar o próximo post do blog: ")
    print()

    # --- Obter os Links dos Produtos do Lojista (Novo) ---
    links_produtos_str = input("🔗 Se você tiver links de produtos de referência (separados por vírgula), digite-os aqui. Caso contrário, apenas pressione Enter: ")
    links_produtos = [link.strip() for link in links_produtos_str.split(',')] if links_produtos_str else None
    print()

    # --- Executar os Agentes ---
    print(" 🔍 Buscando lançamentos relevantes...")
    lancamentos_buscados = agente_buscador(topico, data_de_hoje, links_produtos)
    print(f"Lançamentos encontrados:\n{to_markdown(lancamentos_buscados)}")
    print()

    print(" ✍️ Planejando o post...")
    plano_do_post = agente_planejador(topico, lancamentos_buscados)
    print(f"Plano do post:\n{to_markdown(plano_do_post)}")
    print()

    print(" 📝 Redigindo o rascunho do post...")
    rascunho_gerado = agente_redator(topico, plano_do_post)
    print(f"Rascunho do post:\n{to_markdown(rascunho_gerado)}")
    print()

    print(" 🖼️ Buscando imagens relevantes...")
    imagens_encontradas = agente_buscador_imagens(topico, rascunho_gerado)
    print(f"Imagens encontradas:\n{to_markdown(imagens_encontradas)}")
    print()

    print(" 🧐 Revisando o rascunho...")
    texto_revisado = agente_revisor(topico, rascunho_gerado)
    print(f"Revisão do post:\n{to_markdown(texto_revisado)}")
    print()

    print(" ✨ Formatando o post com imagens...")
    post_formatado = agente_formatador_imagens(rascunho_gerado, imagens_encontradas)
    print("Post formatado (em Markdown):\n")
    print(post_formatado)
