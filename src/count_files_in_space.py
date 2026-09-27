from fetch import list_ai_matcher_files

def main():
    files = list_ai_matcher_files()
    print(f"Nombre total de fichiers sur Digital Ocean : {len(files)}")

if __name__ == "__main__":
    main()
