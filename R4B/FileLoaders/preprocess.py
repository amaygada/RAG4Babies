"""
Service for preprocessing input files
"""

import re
from bs4 import BeautifulSoup
# from R4B.FileLoaders.loader import Loader #UNCOMMENT

# from loader import Loader #COMMENT

class Preprocess:
    def __init__(self, min_content_size=200, min_title_length=5, max_title_length=100, use_removed_list=True, heading_defined_by_font_size=True):
        self.Options = {
            "NaiveText": self.preprocess_naive_text,
            "HTMLParse": self.preprocess_html
        }
        self.max_title_length = max_title_length
        self.use_removed_list = use_removed_list
        self.heading_defined_by_font_size = heading_defined_by_font_size
        self.min_title_length = min_title_length
        self.min_content_size = min_content_size
    
    def preprocess(self, option, content):
        if option in self.Options:
            return self.Options[option](content)
        else:
            raise RuntimeError("Invalid preprocessing option")

    def preprocess_naive_text(self, data):
        """
        Takes as input some text content
        Does the following basic preprocessing on it.
            - remove consecutive '\n' except 1
            - remove consecutive spaces except 1
            - remove the hyphens caused by incomplete words passed on to the next line
        """
        result = []
        for d in data:
            content = d.page_content
            content = re.sub('\n+', ' ', content)
            content = re.sub('- ', '', content)
            content = re.sub('-\n', '', content)
            result.append(content)
        return {"title": "", "entire_doc": result[0]}
    
    def preprocess_html(self, content):
        '''
            Takes as input HTML formatted content and applies bs4 to extract information
            Uses predefined variables to better extract data
        '''
        content = content[0].page_content
        snippets = self.__extract_format_and_content__(content)
        (snippets, removed_list) = self.__process_page_wise__(snippets)
        snippets = self.__create_heading_wise_sections__(snippets, removed_list)
        snippets = self.__coalesce__(snippets)
        
        doc = ""
        for s in snippets:
            doc+=s["content"]

        return {"title": snippets, "entire_doc": doc}

    ############################### HELPER FUNCTIONS ##########################################
    
    def __basic_manipulations__(self, data):
        data = re.sub('\n+', '\n', data)
        data = re.sub('-\n', '', data)
        data = re.sub('- ', '', data)
        return data
    
    def __extract_format_and_content__(self, html_content):
        '''
            Use bs4 to find all divs
            Find all anchors to get page breaks
            Find all divs to get information about "bold" and "font size"
        '''
        soup = BeautifulSoup(html_content,'html.parser')
        content = soup.find_all('div')
        snippets = []
        for c in content:
            a = c.find('a')
            if a:
                regex = re.compile('^Page\s\d+')
                if re.match(regex, c.text):
                    snippets.append((c.text, "a", ""))
            sp = c.find('span')
            if not sp:
                continue
            bold = False
            st = sp.get('style')
            if "bold" in st.lower():
                bold=True
            if not st:
                continue
            fs = re.findall('font-size:(\d+)px',st)
            if not fs:
                continue
            fs = int(fs[0])
            snippets.append((c.text, fs, bold))
        return snippets
    

    def __process_page_wise__(self, snippets):
        '''
            * Divide the content page wise
            * While dividing content page wise, check for and ignore the following:
                * Only numbers (could mean page numbers)
                * Only number with a fullstop (could also mean a page number or chapter number ...)
            * For each page, store top 4 and botom 4 sentences in a set and keep on appending. This is then simultaneously used to remove headers or footers
            * After all the headers are removed, unimportant data is removed, and the data is preprocessed, combine all of them into one for hierarchy mapping.
        '''
        ll = []
        final_list = []
        possible_headers_footers = set()
        removed_list = set()
        curr_page = 1

        for snip in snippets:
            (data, formatting, bold) = snip
            if formatting=='a':
                curr_page = int(data.split(" ")[-1])
                if len(ll)>0:
                    for i in range(0,3):
                        try:
                            ll[i][0]
                        except:
                            break
                        if len(ll[i][0])<150:
                            possible_headers_footers.add(ll[i][0])
                        elif len(ll[-(i)][0])<150:
                            possible_headers_footers.add(ll[-(i)][0])
                ll = []
                continue

            regex = re.compile('^\d+\s*\.*\n*$')
            if re.match(regex, data.strip(' \t\r')):
                continue
            
            #ignore if data already found in the possible list of headers and footers
            if data in possible_headers_footers:
                removed_list.add(data.strip()) if self.heading_defined_by_font_size else removed_list.add(data.strip(' \t\r'))
                continue
            
            #add metadata to data
            metadata = {}
            metadata["bold"] = bold
            metadata["fontsize"] = formatting
            metadata["page"] = curr_page
            
            ll.append((data, metadata))
            final_list.append((data.strip(), metadata)) if self.heading_defined_by_font_size else final_list.append((data.strip(' \t\r'), metadata))

        return (final_list, removed_list)
    

    def __is_heading__(self, snippets, index):
        '''
            Decides wether a snippet is a possible heading based on the following
                * Has font size greater than the immediate content beneath.
                * NEW LINE + BOLD + NEW LINE = HEADING
                * len(data) < Max_title_length  // based on the assumption that title won't be more than a certain amount of characters. Helps eradicate mispredictions.
        '''
        (data, metadata) = snippets[index]
        fs = metadata["fontsize"]
        bold = metadata["bold"]
        if (len(data) > self.max_title_length) or (len(data) < self.min_title_length):
            return False
        if index+1 < len(snippets):
            if snippets[index+1][1]["fontsize"] < fs:
                return True
        if bold:
            if (index-1 >= 0) and (snippets[index-1][0].strip(' \t\r').endswith('\n')) and (data.strip(' \t\r').endswith('\n')):
                return True
            elif (index-1 < 0) and (data.strip(' \t\r').endswith('\n')):
                return True
        return False
    

    def __create_heading_wise_sections__(self, snippets, removed_list):
        '''
            Step 1: If I see a heading
                - I flush the previous data into a list
                - I add the new heading
            Step 2: I wait for Another heading and go back to Step 1    -  else  Go to step 3
            Step 3: wait for the end of the document and append everything in the latest string
        '''
        sections = []
        section = ""
        title = ""
        pages = set()
        
        ll = []

        for i in range(len(snippets)):
            if (self.use_removed_list) and (snippets[i][0] in removed_list):
                continue
            if self.__is_heading__(snippets, i):
                #flush previous data if any to the sections list, and start a new section with this as a heading
                if section!="":
                    dd = {
                            "content": section, 
                            "title": self.__basic_manipulations__(title),
                            "pages": pages
                    }
                    sections.append(dd)
                section = self.__basic_manipulations__(snippets[i][0])
                title = self.__basic_manipulations__(snippets[i][0])
                pages = set()
                pages.add(snippets[i][1]["page"])
            else:
                #add data to current section; we continue the same section and fill it up with more data.
                section += " " + self.__basic_manipulations__(snippets[i][0])
                pages.add(snippets[i][1]["page"])
                if i==len(snippets)-1:
                    sections.append({"content": section, "title": self.__basic_manipulations__(title), "pages": pages})
        return sections
    
    def __coalesce__(self, snippets):
        '''
            Iterates through the snippets
            - checks if the content under each title is sufficient
            - If not it coalesces the current snippet with the one immediately below it
               - The titles are combined
               - The content is also combined
        '''
        i = 0
        while(i<len(snippets)):
            snippet = snippets[i]
            if len(snippet["content"])<self.min_content_size:
                if i+1<len(snippets):
                    snippets[i+1] = {
                        "content": snippet["content"] + " " + snippets[i+1]["content"],
                        "title": snippet["title"] + " " + snippets[i+1]["title"],
                        "pages": snippet["pages"].union(snippets[i+1]["pages"])
                    }
                    snippets.pop(i)
            else:
                i+=1
        return snippets



# l = Loader()
# p = Preprocess()
# p.preprocess("HTMLParse", l.read_html_pdf("p.pdf"))