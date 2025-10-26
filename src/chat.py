from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    try:
        chain = search_prompt()
    except Exception as exc:
        print(
            "Não foi possível iniciar o chat. Verifique as configurações e tente novamente."
        )
        print(f"Detalhes: {exc}")
        return

    print("Faça sua pergunta (digite 'sair' para encerrar).\n")
    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando chat. Até logo!")
            break

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando chat. Até logo!")
            break

        try:
            answer = chain(question)
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}")
            continue

        print(f"RESPOSTA: {answer}\n---\n")


if __name__ == "__main__":
    main()
