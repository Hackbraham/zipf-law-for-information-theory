import zipfslaw


def main():

    print("-----------------")
    print("| Zipf's Law    |")
    print("-----------------\n")

    try:

        with open("dfs/preprocessed-df.csv", "r") as f:
            text = f.read()
            f.close()

            zipf_table = zipfslaw.generate_zipf_table(text, 200)

            zipfslaw.print_zipf_table(zipf_table)

    except IOError as e:

        print(e)


main()
