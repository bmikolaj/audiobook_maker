"""
generate_pdfs.py

Generate one PDF file per string in a list using PyMuPDF (fitz).

Install dependency:
    pip install pymupdf
"""

import fitz  # PyMuPDF
import os


from audiobook_maker import VOICE_INFO
voices = list(set(VOICE_INFO))


def generate_pdfs():
    for voice in voices:
        doc = fitz.open()  # new, empty PDF

        page = doc.new_page()  # default: A4 (595 × 842 pt)
        width, height = page.rect.width, page.rect.height

        margin: int = 50

        # Text bounding box respects all four margins
        text_rect = fitz.Rect(margin, margin, width - margin, height - margin)

        language = "an American" if voice[0] == 'a' else "British"
        gender = "male" if voice[1] == 'm' else "female"
        name = voice.split("_")[1]
        description = VOICE_INFO[voice].split("(")[0].strip()

        text = f"""
                Hello. This is a sample of {language} English {gender} voice.
                My name is {name} and Claude describes me as {description}.
                I am going to say a few sample sentences for your reference.
                
                The quick brown fox jumps over the lazy dog.
                I said THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG! That was all caps this time.
                
                Some other things I might say in audiobooks are:
                Winter is coming.
                {"They're Taking the Hobbits to Isengard!".upper()}
                The answer is 42.
                Well yes, but actually no.
                Elementary my dear Watson.
                
                This concludes the sample of {name}.
                """

        page.insert_textbox(
            text_rect,
            text,
            fontsize=12,
            fontname="helv", # Built-in: helv, tiro, cour, zadb, symb
            color=(0, 0, 0),   # black
            align=fitz.TEXT_ALIGN_LEFT,
        )

        os.makedirs('samples', exist_ok=True)
        output_path = os.path.join('samples', f"{voice}_sample.pdf")
        doc.save(output_path)
        doc.close()

        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    generate_pdfs()
    print(f"\nDone. {len(voices)} file(s) created.")