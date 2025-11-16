"""
Korean-Optimized Tokenizer using KoNLPy

Replaces simple space-based tokenization with morphological analysis
Supports multiple tokenization strategies: Okt, Kkma, Mecab
"""

import re
from typing import List, Dict, Set
from collections import Counter


class KoreanTokenizer:
    """
    Korean text tokenizer with morphological analysis
    
    Automatically detects available KoNLPy backend:
    - Okt (Open Korean Text) - Fast and accurate
    - Kkma - Detailed analysis
    - Mecab - Fastest if available
    """
    
    def __init__(self, backend='okt', lowercase=True, remove_stopwords=True):
        """
        Args:
            backend: 'okt', 'kkma', or 'mecab'
            lowercase: Convert to lowercase
            remove_stopwords: Remove Korean stopwords
        """
        self.backend = backend.lower()
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.tokenizer = self._init_tokenizer()
        
        # Korean stopwords
        self.stopwords = {
            '의', '에', '가', '을', '를', '이', '고', '도', '하', '는', '은',
            '와', '로', '으로', '하고', '하면', '하면서', '하거나', '하고나',
            '있고', '있으면', '그리고', '그러면', '그러나', '그래도', '그런데',
            '다', '다만', '다시', '다시다', '다시말해', '다소', '대로', '대신',
            '당신', '당신이', '당신의', '대체', '더', '더군다나', '더라도', '더욱',
            '더없이', '더이상', '더욱', '데', '도', '도록', '도움', '도움이',
            '돈', '되', '되고', '되겠', '되겠네', '되겠습니다', '되겠어',
            '수', '수도', '수도있고', '수도있으며', '수도있지', '수없',
            '수없고', '수없으며', '수없지', '것', '것과', '것도', '것들',
            '것이', '것이다', '것이지', '것처럼', '것처럼'
        }
        
        print(f"KoreanTokenizer initialized:")
        print(f"  Backend: {self.backend}")
        print(f"  Lowercase: {lowercase}")
        print(f"  Remove stopwords: {remove_stopwords}")
    
    def _init_tokenizer(self):
        """Initialize KoNLPy tokenizer"""
        try:
            if self.backend == 'okt':
                try:
                    from konlpy.tag import Okt
                    return Okt()
                except Exception as e:
                    print(f"Warning: Okt not available ({e}). Falling back to Kkma.")
                    self.backend = 'kkma'
            
            if self.backend == 'kkma':
                try:
                    from konlpy.tag import Kkma
                    return Kkma()
                except Exception as e:
                    print(f"Warning: Kkma not available ({e}). Falling back to Mecab.")
                    self.backend = 'mecab'
            
            if self.backend == 'mecab':
                try:
                    from konlpy.tag import Mecab
                    return Mecab()
                except Exception as e:
                    print(f"Warning: Mecab not available ({e}). Using fallback tokenizer.")
                    return None
        
        except ImportError:
            print("Warning: KoNLPy not installed. Install with: pip install konlpy jpype1")
            print("Falling back to simple tokenizer.")
            return None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Korean text with morphological analysis
        
        Args:
            text: Input Korean text
        
        Returns:
            List of tokens
        """
        # Clean text
        text = self._clean_text(text)
        
        if self.tokenizer is None:
            # Fallback to simple tokenization
            return self._tokenize_simple(text)
        
        try:
            if self.backend == 'okt':
                tokens = self.tokenizer.morphs(text)
            elif self.backend == 'kkma':
                tokens = self.tokenizer.morphs(text)
            elif self.backend == 'mecab':
                tokens = self.tokenizer.morphs(text)
            else:
                tokens = self._tokenize_simple(text)
        
        except Exception as e:
            print(f"Tokenization error: {e}. Using fallback tokenizer.")
            tokens = self._tokenize_simple(text)
        
        # Post-processing
        tokens = [t for t in tokens if len(t) > 0]
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        return tokens
    
    def _clean_text(self, text: str) -> str:
        """Clean text: remove special characters, normalize spacing"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        # Keep only Korean, English, numbers, and basic punctuation
        text = re.sub(r'[^\uac00-\ud7afa-zA-Z0-9\s\.,!?\-()]', '', text)
        
        return text.strip()
    
    def _tokenize_simple(self, text: str) -> List[str]:
        """Simple fallback tokenizer (when KoNLPy unavailable)"""
        # Split by spaces and punctuation
        tokens = re.split(r'[\s\.,!?\-()]+', text)
        tokens = [t for t in tokens if len(t) > 0]
        return tokens


class KoreanVocabulary:
    """Korean vocabulary with better tokenization and word frequency tracking"""
    
    def __init__(self, min_freq: int = 2, tokenizer_backend: str = 'okt'):
        """
        Args:
            min_freq: Minimum word frequency to include in vocabulary
            tokenizer_backend: 'okt', 'kkma', or 'mecab'
        """
        self.min_freq = min_freq
        self.tokenizer = KoreanTokenizer(backend=tokenizer_backend)
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # Token to index
        self.stoi = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        
        # Index to token
        self.itos = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        self.freq = Counter()
        self.pad_idx = 0
        self.unk_idx = 1
    
    def build(self, texts: List[str], max_vocab_size: int = 50000):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size (excluding special tokens)
        """
        print(f"Building vocabulary from {len(texts)} texts...")
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            self.freq.update(tokens)
        
        print(f"Found {len(self.freq)} unique tokens")
        
        # Sort by frequency
        sorted_tokens = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        
        # Add tokens to vocabulary
        count = 0
        for token, freq in sorted_tokens:
            if freq >= self.min_freq and count < max_vocab_size:
                if token not in self.stoi:
                    idx = len(self.itos)
                    self.stoi[token] = idx
                    self.itos.append(token)
                    count += 1
        
        print(f"Vocabulary size: {len(self.itos)} tokens")
        print(f"  - Special tokens: 4")
        print(f"  - Regular tokens: {len(self.itos) - 4}")
    
    def encode(self, text: str, max_len: int = 64, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            max_len: Maximum sequence length (truncation)
            add_special_tokens: Add BOS and EOS tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.tokenize(text)
        
        # Truncate
        tokens = tokens[:max_len]
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Convert to IDs
        token_ids = [self.stoi.get(t, self.unk_idx) for t in tokens]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        tokens = [self.itos[idx] if idx < len(self.itos) else self.unk_token for idx in token_ids]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in {self.pad_token, self.unk_token, self.bos_token, self.eos_token}]
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.itos)
    
    def get_stats(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'vocab_size': len(self.itos),
            'unique_tokens': len(self.freq),
            'min_freq': self.min_freq,
            'total_token_occurrences': sum(self.freq.values()),
            'average_freq': sum(self.freq.values()) / len(self.freq) if self.freq else 0
        }


if __name__ == '__main__':
    print("Testing Korean Tokenizer & Vocabulary")
    print("=" * 60)
    
    # Example Korean texts
    test_texts = [
        "왼쪽 상단 모서리에 있는 빨간색 버튼을 찾으세요",
        "화면 중앙의 입력 필드에 텍스트를 입력하세요",
        "하단 우측 코너에서 저장 버튼을 클릭하세요",
        "상단 네비게이션 바에서 메뉴를 선택하세요",
        "텍스트 입력 박스 옆의 제출 버튼을 누르세요"
    ]
    
    # Test tokenizer
    print("\n1. Testing KoreanTokenizer:")
    print("-" * 60)
    tokenizer = KoreanTokenizer(backend='okt', remove_stopwords=True)
    
    for text in test_texts[:2]:
        tokens = tokenizer.tokenize(text)
        print(f"Original: {text}")
        print(f"Tokens:   {tokens}")
        print()
    
    # Test vocabulary
    print("2. Testing KoreanVocabulary:")
    print("-" * 60)
    vocab = KoreanVocabulary(min_freq=1, tokenizer_backend='okt')
    vocab.build(test_texts)
    
    print("\nVocab Stats:")
    stats = vocab.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Encoding/Decoding
    print("\n3. Testing Encoding/Decoding:")
    print("-" * 60)
    test_text = test_texts[0]
    
    # Encode
    token_ids = vocab.encode(test_text, max_len=64, add_special_tokens=True)
    print(f"Original text: {test_text}")
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded = vocab.decode(token_ids)
    print(f"Decoded text: {decoded}")
    
    print("\n✓ All tests completed!")
