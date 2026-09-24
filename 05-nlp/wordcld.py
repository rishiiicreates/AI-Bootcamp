import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')


text = """
I’m Hrishikesh—though if you know me at all, it’s just Rishii. I’m an 18-year-old Computer Science student at SRMIST, constantly navigating that strange, thin line between hard engineering logic and midnight raw thought. Half my time is spent building systems—crafting software like Nudge, Housel, and DashMetrics, turning chaotic ideas into clean code on my M5 Mac while figuring out how the world actually works behind the screen. The other half belongs to 3 AM, where I write poetry and prose about the exact things people feel but are far too polite or scared to say out loud. My writing doesn't exist to comfort or soften the edges of reality; it exists to reflect truth, no matter how harsh or unvarnished it gets. Standing at 6'1" and around 78 kg, I balance the relentless desk hours with bodyweight training and heavy pull-up sessions—mostly to keep the chaotic energy grounded. I’m quietly competitive, fiercely self-aware, and entirely unimpressed by empty ambition or dressed-up lazy plans. I see the absurdity in failure, laugh at my own flaws first so nobody else can hold them, and keep moving forward. Whether I'm engineering tech, analyzing data, or writing lines that cut right to the nerve, I’m just here to build, see clearly, and create things that endure """

words = word_tokenize(text.lower())  # Convert to lowercase for uniformity
print(words)


stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
# The general syntax for a list comprehension is: [expression for item in iterable if condition]
print(filtered_words)

word_freq = Counter(filtered_words)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()
