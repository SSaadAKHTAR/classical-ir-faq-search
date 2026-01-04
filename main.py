import re
import math
import time
import json
from collections import defaultdict, Counter

class InteractiveDocumentQA:
    """Interactive Command-Line Document Q&A System with Step-by-Step Guidance"""
    
    def __init__(self, k1=1.5, b=0.75):
        # Document storage
        self.documents = []
        self.sentences = []
        self.paragraphs = []
        
        # IR components
        self.word_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.idf = {}
        self.tf_idf_vectors = []
        
        # BM25 parameters
        self.k1 = k1
        self.b = b
        self.avg_doc_length = 0
        self.doc_lengths = []
        
        self.stopwords = self._get_stopwords()
        self.document_loaded = False
        
    def _get_stopwords(self):
        """Common English stopwords"""
        return set([
            'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by',
            'can', 'do', 'for', 'from', 'has', 'have', 'he', 'her', 'his',
            'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'this',
            'to', 'was', 'were', 'will', 'with', 'would', 'could', 'should'
        ])
    
    def print_header(self, text):
        """Print styled header"""
        print("\n" + "="*70)
        print(text.center(70))
        print("="*70)
    
    def print_section(self, text):
        """Print section header"""
        print("\n" + "-"*70)
        print(text)
        print("-"*70)
    
    def print_step(self, step_num, description):
        """Print step indicator"""
        print(f"\n[STEP {step_num}] {description}")
    
    def loading_animation(self, message, duration=1.5):
        """Show loading animation"""
        import sys
        chars = ["-", "\\", "|", "/"]
        end_time = time.time() + duration
        i = 0
        while time.time() < end_time:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        print(f"\r[+] {message}... Done!   ")
    
    # ==================== TEXT PROCESSING ====================
    
    def tokenize(self, text):
        """Tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b[\w]+\b', text)
        return tokens
    
    def preprocess(self, tokens):
        """Preprocessing: stopword removal and stemming"""
        filtered = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        stemmed = []
        for word in filtered:
            if word.endswith('ies') and len(word) > 4:
                word = word[:-3] + 'y'
            elif word.endswith('es') and len(word) > 3:
                word = word[:-2]
            elif word.endswith('ed') and len(word) > 3:
                word = word[:-2]
            elif word.endswith('ing') and len(word) > 4:
                word = word[:-3]
            elif word.endswith('s') and len(word) > 3:
                word = word[:-1]
            stemmed.append(word)
        
        return stemmed
    
    def split_into_sentences(self, text):
        """Sentence segmentation"""
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def split_into_paragraphs(self, text):
        """Paragraph segmentation"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        return paragraphs
    
    # ==================== INDEXING ====================
    
    def load_document(self, document_text):
        """Load and index document with progress updates"""
        print("\n[INFO] Processing your document...")
        
        self.documents = [document_text]
        
        self.loading_animation("Segmenting into sentences", 0.5)
        self.sentences = self.split_into_sentences(document_text)
        print(f"   Found {len(self.sentences)} sentences")
        
        self.loading_animation("Segmenting into paragraphs", 0.5)
        self.paragraphs = self.split_into_paragraphs(document_text)
        print(f"   Found {len(self.paragraphs)} paragraphs")
        
        self.loading_animation("Building inverted index", 0.8)
        self._build_inverted_index()
        
        self.loading_animation("Calculating TF-IDF weights", 0.6)
        self._calculate_idf()
        self._build_tfidf_vectors()
        
        self.document_loaded = True
        print("\n[SUCCESS] Document indexed successfully!")
        print(f"   Total words: {len(self.tokenize(document_text))}")
        print(f"   Unique terms: {len(self.idf)}")
        
    def _build_inverted_index(self):
        """Build inverted index"""
        self.word_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        self.doc_lengths = []
        
        for idx, sentence in enumerate(self.sentences):
            tokens = self.tokenize(sentence)
            processed = self.preprocess(tokens)
            
            self.doc_lengths.append(len(processed))
            
            term_set = set(processed)
            for term in term_set:
                self.word_index[term].append(idx)
                self.doc_freq[term] += 1
        
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
    
    def _calculate_idf(self):
        """Calculate IDF values"""
        N = len(self.sentences)
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((N + 1) / (df + 1))
    
    def _build_tfidf_vectors(self):
        """Build TF-IDF vectors"""
        self.tf_idf_vectors = []
        
        for sentence in self.sentences:
            tokens = self.tokenize(sentence)
            processed = self.preprocess(tokens)
            
            tf_counter = Counter(processed)
            
            vector = {}
            for term, tf in tf_counter.items():
                vector[term] = tf * self.idf.get(term, 0)
            
            self.tf_idf_vectors.append(vector)
    
    # ==================== RETRIEVAL ====================
    
    def bm25_score(self, query_terms, doc_idx):
        """Calculate BM25 score"""
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        
        sentence = self.sentences[doc_idx]
        tokens = self.tokenize(sentence)
        processed = self.preprocess(tokens)
        tf_counter = Counter(processed)
        
        for term in query_terms:
            if term not in tf_counter:
                continue
            
            tf = tf_counter[term]
            idf = self.idf.get(term, 0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve_passages(self, query, top_k=5, verbose=True):
        """Retrieve top-k relevant passages"""
        if verbose:
            print(f"\n[RETRIEVAL] Searching for relevant passages...")
        
        query_tokens = self.tokenize(query)
        query_terms = self.preprocess(query_tokens)
        
        if verbose:
            print(f"   Query terms: {query_terms[:10]}{'...' if len(query_terms) > 10 else ''}")
        
        scores = []
        for idx in range(len(self.sentences)):
            score = self.bm25_score(query_terms, idx)
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append({
                    'sentence': self.sentences[idx],
                    'score': score,
                    'index': idx
                })
        
        if verbose:
            print(f"   [+] Found {len(results)} relevant passages")
        
        return results
    
    # ==================== READING ====================
    
    def _classify_question_type(self, question):
        """Classify question type"""
        question_lower = question.lower().strip()
        
        if question_lower.startswith('who'):
            return 'who'
        elif question_lower.startswith('when'):
            return 'when'
        elif question_lower.startswith('where'):
            return 'where'
        elif question_lower.startswith('how many') or question_lower.startswith('how much'):
            return 'how_many'
        elif question_lower.startswith('what'):
            return 'what'
        elif question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('how'):
            return 'how'
        else:
            return 'general'
    
    def read_and_extract_answer(self, question, passages, verbose=True):
        """Extract precise answer from passages"""
        if verbose:
            print(f"\n[READING] Analyzing passages to extract answer...")
        
        if not passages:
            return {
                'answer': "No relevant information found in the document.",
                'confidence': 0.0,
                'source_passage': None,
                'answer_type': 'none'
            }
        
        question_type = self._classify_question_type(question)
        if verbose:
            print(f"   Question type detected: {question_type.upper()}")
        
        best_passage = passages[0]
        text = best_passage['sentence']
        
        # Extract based on question type
        if question_type == 'what':
            match = re.search(r'(\w+(?:\s+\w+){0,3})\s+(is|are)\s+(.+)', text.lower())
            if match:
                return {
                    'answer': text,
                    'confidence': 0.85,
                    'source_passage': text,
                    'answer_type': 'definition'
                }
        
        elif question_type == 'when':
            year_match = re.search(r'\b(19|20)\d{2}\b', text)
            if year_match:
                return {
                    'answer': year_match.group(0),
                    'confidence': 0.9,
                    'source_passage': text,
                    'answer_type': 'time'
                }
        
        elif question_type == 'how_many':
            number_match = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', text.lower())
            if number_match:
                return {
                    'answer': number_match.group(0),
                    'confidence': 0.9,
                    'source_passage': text,
                    'answer_type': 'number'
                }
        
        return {
            'answer': text,
            'confidence': 0.7,
            'source_passage': text,
            'answer_type': question_type
        }
    
    def answer_question(self, question, verbose=False):
        """Complete Q&A pipeline"""
        if not self.document_loaded:
            print("\n[ERROR] No document loaded. Please load a document first!")
            return None
        
        # Verbose flag overrides for silent running if needed, but we keep UI minimal
        
        # Phase 1: Retrieval
        passages = self.retrieve_passages(question, top_k=5, verbose=False)
        
        if not passages:
            print("\n[!] No relevant information found.")
            return None
        
        # Phase 2: Reading
        result = self.read_and_extract_answer(question, passages, verbose=False)
        
        # Simplified Output
        print("\n" + "-"*70)
        print(f"Answer: {result['answer']}")
        print("-"*70)
        
        return result
    
    # ==================== FAQ GENERATION ====================
    
    def generate_faq(self, num_questions=5):
        """Generate FAQs from document"""
        print("\n[FAQ GENERATION] Generating frequently asked questions...")
        
        sentence_scores = {}
        for idx, vector in enumerate(self.tf_idf_vectors):
            score = sum(vector.values())
            
            sentence_lower = self.sentences[idx].lower()
            if any(word in sentence_lower for word in ['important', 'key', 'main', 'significant']):
                score *= 1.5
            
            if len(self.sentences[idx].split()) < 5:
                score *= 0.5
            
            sentence_scores[idx] = score
        
        # Increase candidate pool to improve chances of finding good questions
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_questions*5]
        
        faqs = []
        for idx, score in top_sentences:
            sentence = self.sentences[idx]
            question = self._generate_question(sentence)
            
            if question:
                answer = sentence
                faqs.append({
                    'question': question,
                    'answer': answer,
                    'score': score
                })
            
            if len(faqs) >= num_questions:
                break
        
        print(f"   [+] Generated {len(faqs)} questions")
        return faqs
    
    def _generate_question(self, sentence):
        """Generate question from sentence"""
        sentence_lower = sentence.lower()
        
        # Try to find a definition pattern: "X is Y", "X are Y", "X refers to Y"
        # We use a more permissive patter to capture the subject
        definition_match = re.search(r'^([\w\s\-]+(?:\s+\w+){0,5})\s+(?:is|are|refers to|means)\s+', sentence_lower)
        if definition_match:
            subject = definition_match.group(1).strip()
            # Avoid questions where subject is too long or looks like a partial sentence
            if len(subject.split()) <= 6 and len(subject) > 2:
                # Basic heuristics to avoid bad subjects like "it", "this", "they" (demonstrative pronouns)
                if subject.lower() not in ['it', 'this', 'that', 'they', 'these', 'those', 'there', 'here']:
                    return f"What is {subject}?"
        
        # Try to find a causal/explanation pattern: "because", "due to", "as a result"
        if any(marker in sentence_lower for marker in ['because', 'due to', 'leads to', 'results in', 'in order to']):
             # If it's a "because" clause, we can try to turn the first part into a "Why" question
             if 'because' in sentence_lower:
                parts = sentence_lower.split('because')
                prefix = parts[0].strip()
                if len(prefix) > 10:
                    # Very naive attempt to convert statement to question
                    # Ideally we'd use NLP, but here we just prepend "Why"
                    return f"Why {prefix}?"
             else:
                 # Generic why question based on important keywords could be better, 
                 # but for now let's fall back to "How/Why" on the whole sentence topic.
                 pass

        # Fallback for high-value sentences: Generate a generic question based on the first noun phrase or main subject
        # simpler fallback: "Tell me more about..." or "What is the significance of..."
        # We can extract the first few words as a proxy for the topic if it looks like a noun phrase
        
        words = sentence.split()
        if len(words) > 3:
            # Check if first few words are capitalized (likely a proper noun or title start) 
            # or just use the first 2-3 words as a loose topic
            topic = ' '.join(words[:3])
            # pattern check to ensure we don't end with a weird word
            if topic.lower() not in ['however,', 'moreover,', 'furthermore,', 'in addition,']:
                 return f"What can you tell me about {topic}?"

        return None
    
    def print_faq(self, faqs):
        """Print FAQs in a nice format"""
        self.print_header("FREQUENTLY ASKED QUESTIONS")
        
        for i, faq in enumerate(faqs, 1):
            print(f"\n{i}. Q: {faq['question']}")
            print(f"   A: {faq['answer'][:150]}{'...' if len(faq['answer']) > 150 else ''}")
            print("-"*70)
    
    # ==================== INTERACTIVE MENU ====================
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "="*70)
        print("INTERACTIVE DOCUMENT Q&A SYSTEM - MAIN MENU".center(70))
        print("="*70)
        print("\n1. Load Document")
        print("2. Ask a Question")
        print("3. Generate FAQs")
        print("4. View Document Statistics")
        print("5. Show Example Questions")
        print("6. Exit")
        print("\n" + "="*70)
    
    def show_statistics(self):
        """Show document statistics"""
        if not self.document_loaded:
            print("\n[ERROR] No document loaded yet!")
            return
        
        self.print_header("DOCUMENT STATISTICS")
        print(f"\nSentences: {len(self.sentences)}")
        print(f"Paragraphs: {len(self.paragraphs)}")
        print(f"Unique Terms: {len(self.idf)}")
        print(f"Average Sentence Length: {self.avg_doc_length:.1f} terms")
        
        # Top terms by IDF
        top_terms = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 Most Important Terms (by IDF):")
        for i, (term, idf_val) in enumerate(top_terms, 1):
            print(f"   {i}. {term} (IDF: {idf_val:.2f})")
    
    def show_example_questions(self):
        """Show example questions"""
        self.print_header("EXAMPLE QUESTIONS YOU CAN ASK")
        print("\nFactoid Questions (WHO, WHAT, WHEN, WHERE):")
        print("   * What is machine learning?")
        print("   * When was AI created?")
        print("   * Who invented the computer?")
        print("   * Where is it used?")
        
        print("\nQuantitative Questions (HOW MANY):")
        print("   * How many types of machine learning are there?")
        print("   * How many applications are mentioned?")
        
        print("\nProcess Questions (HOW, WHY):")
        print("   * How does supervised learning work?")
        print("   * Why is machine learning important?")
    
    def load_document_interactive(self):
        """Interactive document loading"""
        self.print_step(1, "LOAD YOUR DOCUMENT")
        
        print("\nChoose an option:")
        print("1. Use sample document (Machine Learning)")
        print("2. Paste your own document")
        print("3. Load from file")
        print("4. Load from JSON file (FAQ format)")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == '1':
            sample_doc = """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Machine learning focuses on developing computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly.

There are three main types of machine learning. Supervised learning uses labeled training data to learn the mapping function from input to output. Unsupervised learning finds hidden patterns or intrinsic structures in input data without labeled responses. Reinforcement learning learns by interacting with an environment and receiving rewards or penalties.

Machine learning algorithms are used in a wide variety of applications. Email filtering and computer vision are common examples where it is difficult or infeasible to develop conventional algorithms. Machine learning is closely related to computational statistics, which focuses on making predictions using computers.

Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks attempt to simulate the behavior of the human brain to learn from large amounts of data. While a neural network with a single layer can make approximate predictions, additional hidden layers can help optimize accuracy.

Common applications of machine learning include recommendation systems used by Netflix and Amazon, fraud detection in banking, speech recognition in virtual assistants, autonomous vehicles, medical diagnosis, and natural language processing. The field continues to grow rapidly with new applications emerging constantly."""
            self.load_document(sample_doc)
            
        elif choice == '2':
            print("\nPaste your document below (press Enter, then Ctrl+D on Unix or Ctrl+Z on Windows when done):")
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            document = '\n'.join(lines)
            if document.strip():
                self.load_document(document)
            else:
                print("\n[ERROR] No document provided!")
                
        elif choice == '3':
            filename = input("\nEnter filename: ").strip()
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    document = f.read()
                self.load_document(document)
            except FileNotFoundError:
                print(f"\n[ERROR] File '{filename}' not found!")
            except Exception as e:
                print(f"\n[ERROR] Error reading file: {e}")
        elif choice == '4':
            self._load_from_json()
        else:
            print("\n[ERROR] Invalid choice!")
    
    def _load_from_json(self):
        """Load document from a JSON file in FAQ format."""
        filename = input("\nEnter JSON filename: ").strip()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract answers and concatenate them
            document = "\n".join([item['answer'] for item in data if 'answer' in item])
            
            if document.strip():
                self.load_document(document)
            else:
                print("\n[ERROR] No valid answers found in the JSON file!")

        except FileNotFoundError:
            print(f"\n[ERROR] File '{filename}' not found!")
        except json.JSONDecodeError:
            print(f"\n[ERROR] Invalid JSON format in '{filename}'!")
        except Exception as e:
            print(f"\n[ERROR] Error reading file: {e}")
    
    def run_interactive(self):
        """Main interactive loop"""
        self.print_header("RETRIEVE-AND-READ DOCUMENT Q&A SYSTEM")
        print("\nWelcome! This system uses a two-phase architecture:")
        print("  1. Retrieval - Find relevant passages")
        print("  2. Reading - Extract precise answers")
        print("\nLet's get started!")
        

        # Force document loading at startup
        # Force document loading at startup
        try:
            while not self.document_loaded:
                print("\n[!] To use this system, you must first load a document.")
                self.load_document_interactive()
                
                if not self.document_loaded:
                    retry = input("\nDo you want to try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("\nExiting system.")
                        return
    
            while True:
                self.show_menu()
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    self.load_document_interactive()
                    
                elif choice == '2':
                    if not self.document_loaded:
                        print("\n[ERROR] Please load a document first (Option 1)!")
                    else:
                        while True:
                            question = input("\nEnter your question: ").strip()
                            if question:
                                self.answer_question(question)
                            else:
                                print("\n[!] Please enter a valid question!")
                            
                            cont = input("\n[?] Do you want to ask another question? (y/n): ").strip().lower()
                            if cont != 'y':
                                break
                            
                elif choice == '3':
                    if not self.document_loaded:
                        print("\n[ERROR] Please load a document first (Option 1)!")
                    else:
                        num = input("\nHow many FAQs to generate? (default: 5): ").strip()
                        num = int(num) if num.isdigit() else 5
                        faqs = self.generate_faq(num)
                        self.print_faq(faqs)
                        
                elif choice == '4':
                    self.show_statistics()
                    
                elif choice == '5':
                    self.show_example_questions()
                    
                elif choice == '6':
                    print("\nThank you for using the Document Q&A System!")
                    print("="*70)
                    break
                    
                else:
                    print("\n[ERROR] Invalid choice! Please enter 1-6.")
                
                input("\nPress Enter to continue...")
        
        except (EOFError, KeyboardInterrupt):
            print("\n\n[INFO] Exiting system.")
            return


# ==================== MAIN FUNCTION ====================

def main():
    """Main function to run the interactive system"""
    qa_system = InteractiveDocumentQA(k1=1.5, b=0.75)
    qa_system.run_interactive()


if __name__ == "__main__":
    main()