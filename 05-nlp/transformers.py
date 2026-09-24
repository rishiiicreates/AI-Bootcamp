sentence1 = ["the", "robber", "went", "to", "the", "bank"]
sentence2 = ["the", "river", "bank", "was", "very", "muddy"]


context_clues = {"robber", "steal", "money", "river", "water", "muddy"}

def calculate_attention(sentence):
    print(f"\nAnalyzing Sentence: {' '.join(sentence)}")
    print("-" * 40)
    
    # 3. Look at every word, and see how much it "pays attention" to other words
    for target_word in sentence:
        attention_score = 0
        connected_words = []
        
        if target_word == "bank":
            for other_word in sentence:
                if other_word in context_clues:
                    attention_score += 1
                    connected_words.append(other_word)
            
            print(f"🎯 Word: '{target_word}'")
            print(f"   ↳ Paid Attention to: {connected_words}")
            print(f"   ↳ Attention Score: {attention_score} (This means it's a financial bank!)" if "robber" in connected_words else f"   ↳ Attention Score: {attention_score} (This means it's nature/land!)")

calculate_attention(sentence1)
calculate_attention(sentence2)
