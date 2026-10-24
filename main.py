"""Entry point for the framework-free Gemini agent."""

from dotenv import load_dotenv

from demo_agent import run

MODEL = "gemini-2.0-flash"


def main():
    load_dotenv()
    run(model=MODEL)


if __name__ == "__main__":
    main()
