import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.preprocessing.preprocessor import TextPreprocessor

def test():
    print("Testing preprocessor...")
    
    # إنشاء المعالج
    preprocessor = TextPreprocessor(use_stemming=False)
    
    # نص اختبار
    test_text = "This is a test document about cloud storage and backup solutions for enterprise data centers"
    
    # معالجة
    tokens = preprocessor.process(test_text)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    if len(tokens) > 0:
        print("✅ Preprocessor works correctly!")
        return True
    else:
        print("❌ Preprocessor returned empty tokens")
        return False

if __name__ == "__main__":
    test()