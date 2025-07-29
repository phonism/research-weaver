"""
Citation management system for research reports
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class Citation:
    """
    A single citation reference
    """
    number: int
    title: str
    date: str
    url: str
    domain: str


class CitationManager:
    """
    Manages citations and converts between footnote and inline formats
    """
    
    def __init__(self):
        self.citations: Dict[int, Citation] = {}
        
    def extract_citations_from_text(self, text: str) -> Tuple[str, Dict[int, Citation]]:
        """
        Extract citations from text with footnote format
        
        Args:
            text: Text with footnote citations and reference section
            
        Returns:
            Tuple of (main_text, citations_dict)
        """
        # Find the markdown reference section
        ref_pattern = r'## 参考文献\s*\n((?:\d+\.\s+\*\*.*(?:\n|$))*)'
        ref_match = re.search(ref_pattern, text, re.MULTILINE | re.DOTALL)
        
        if not ref_match:
            return text, {}
            
        # Extract main text (everything before references)
        main_text = text[:ref_match.start()].strip()
        ref_section = ref_match.group(1)
        
        # Parse individual references
        citations = {}
        ref_lines = ref_section.strip().split('\n')
        
        import logging
        logging.debug(f"[Citation] Reference section: {repr(ref_section)}")
        logging.debug(f"[Citation] Split into {len(ref_lines)} lines")
        
        for i, line in enumerate(ref_lines):
            line = line.strip()
            logging.debug(f"[Citation] Line {i}: {repr(line)}")
            if not line:
                continue
                
            # Parse markdown format: 1. **Title** - Date - [domain.com](url)
            citation_match = re.match(r'(\d+)\.\s+\*\*(.*?)\*\*\s*-\s*(.*?)\s*-\s*\[(.*?)\]\((.*?)\)', line)
            logging.debug(f"[Citation] Citation match for line {i}: {citation_match is not None}")
            
            if citation_match:
                number = int(citation_match.group(1))
                title = citation_match.group(2).strip()
                date = citation_match.group(3).strip()
                domain = citation_match.group(4).strip()
                url = citation_match.group(5).strip()
                    
                citations[number] = Citation(
                    number=number,
                    title=title,
                    date=date,
                    url=url,
                    domain=domain
                )
                
        return main_text, citations
        
    def convert_to_inline_citations(self, text: str) -> str:
        """
        Convert footnote citations to inline format
        
        Args:
            text: Text with footnote format citations
            
        Returns:
            Text with inline format citations
        """
        main_text, citations = self.extract_citations_from_text(text)
        
        if not citations:
            return text
            
        # Replace [1], [2], etc. with [domain.com]
        def replace_citation(match):
            number = int(match.group(1))
            if number in citations:
                citation = citations[number]
                return f" [{citation.domain}]"
            return match.group(0)
            
        # Replace footnote numbers with domain names
        inline_text = re.sub(r'\[(\d+)\]', replace_citation, main_text)
        
        return inline_text.strip()
        
    def get_citation_details(self, text: str) -> List[Citation]:
        """
        Get list of all citations from text
        
        Args:
            text: Text with citations
            
        Returns:
            List of Citation objects
        """
        _, citations = self.extract_citations_from_text(text)
        return list(citations.values())
        
    def add_citation_tracking_to_context(self, researcher_context, search_results: List[Dict]):
        """
        Add citation tracking information to researcher context
        
        Args:
            researcher_context: Researcher's context object
            search_results: List of search results with title, url, date
        """
        citation_info = "\n\n--- 可用引用源 ---\n"
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Unknown Title')
            url = result.get('url', '')
            date = result.get('date', '')
            domain = urlparse(url).netloc if url else 'unknown'
            
            citation_info += f"[{i}] {title} - {date} - {domain}\n"
            
        return citation_info


def format_citations_for_display(text: str) -> Tuple[str, List[Dict]]:
    """
    Format citations for UI display
    
    Args:
        text: Text with footnote citations
        
    Returns:
        Tuple of (inline_text, reference_list)
    """
    manager = CitationManager()
    
    # Convert to inline format
    inline_text = manager.convert_to_inline_citations(text)
    
    # Get citation details for reference list
    citations = manager.get_citation_details(text)
    
    # Format for UI
    reference_list = []
    for citation in citations:
        reference_list.append({
            'number': citation.number,
            'title': citation.title,
            'date': citation.date,
            'url': citation.url,
            'domain': citation.domain
        })
        
    return inline_text, reference_list