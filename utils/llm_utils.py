
from IPython.display import display, HTML

def format_llm_answer(answer: str) -> str:
        """
        Formats a text answer from an LLM into a user-friendly HTML format.

        Args:
            answer (str): The raw text from the LLM.

        Returns:
            str: A formatted HTML version of the answer.
        """
        # Split the text into sentences
        sentences = answer.split('. ')
        
        # Start creating the HTML structure
        html_output = "<div style='font-family: Arial, sans-serif; line-height: 1.6; font-size: 16px; color: #FFFFFF;'>\n"
        html_output += "  <ul style='margin: 10px 0; padding-left: 20px;'>\n"

        # Add each sentence as a bullet point
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Skip empty sentences
                html_output += f"    <li>{sentence.capitalize()}.</li>\n"

        # Close the HTML structure
        html_output += "  </ul>\n"
        html_output += "</div>"

        return display(HTML(html_output))