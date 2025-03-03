from process_research_paper import PDFCleaner

def test_cleaning():
    # Initialize the cleaner
    cleaner = PDFCleaner()
    
    # Test cases with various forms of scientific terms
    test_cases = [
        "HER-2/neu amplification in breast cancer",
        "HER2/neu positive tumors",
        "HER 2/neu overexpression",
        "Her-2/neu negative cases",
        "HER2/neu amplification and survival",
        "proto oncogene expression",
        "hormonal receptor status",
        "erb B expression",
        "HER-2/neu amplification in HER2/neu positive tumors",
        "HER-2/neu overexpression in HER2/neu negative cases"
    ]
    
    print("Testing PDF cleaning process...")
    print("\nTest cases and their cleaned versions:")
    print("-" * 80)
    
    for test in test_cases:
        cleaned = cleaner.clean_text(test)
        print(f"\nOriginal: {test}")
        print(f"Cleaned:  {cleaned}")
        print("-" * 80)

if __name__ == "__main__":
    test_cleaning() 