#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: tools/latex_report.py
Location: tools/
====================================
LaTeX Report Generator (LatexReport)

Purpose:
- Generate academic paper-quality LaTeX documents for research publications
- Support automatic section structuring with configurable document layout
- Enable easy generation of technical reports with proper formatting

Key Features:
1. Automatic Document Structure: Pre-defined academic paper sections and hierarchy
2. Flexible Formatting: Configurable document class options and packages
3. Author Management: Proper author formatting with affiliations
4. Appendix Support: Automatic appendix section generation
5. Bibliography Integration: Built-in bibliography and reference support
6. Multi-format Output: Generate both .tex source and compiled PDF files
"""

from pylatex import Document, Section, Subsection, Subsubsection, Command, Package
from pylatex.utils import NoEscape, italic

def create_paper_template():
    # 1. Initialize document with two-column layout
    doc = Document(documentclass='article', document_options=['twocolumn', '10pt', 'a4paper'])

    # 2. Add essential LaTeX packages (following common paper formats)
    packages = [
        'geometry',  # Page margins
        'graphicx',  # Images
        'amsmath',  # Mathematical formulas
        'hyperref',  # Hyperlinks
        'titlesec',  # Title formatting
        'authblk'  # Author typesetting
    ]
    for pkg in packages:
        doc.packages.append(Package(pkg))

    # Set page margins
    doc.preamble.append(Command('geometry', arguments='margin=0.8in'))

    # 3. Define article structure data (most efficient part, modify here for future changes)
    structure = [
        {"title": "Introduction/Background/Motivation", "type": "sec"},
        {"title": "Approach", "type": "sec", "content": [
            {"title": "Embedding Layer + Positional Encoding", "type": "subsec"},
            {"title": "Spatio-temporal Attention Layer", "type": "subsec", "content": [
                {"title": "Temporal Attention Layer", "type": "subsubsec"},
                {"title": "Spatial Attention Layer", "type": "subsubsec"},
                {"title": "Feed forward layer", "type": "subsubsec"},
            ]},
            {"title": "Output Projection Layer", "type": "subsec"},
            {"title": "Residual Connections and LayerNorm", "type": "subsec"},
            {"title": "Teacher/Student & Curriculum Learning", "type": "subsec"},
        ]},
        {"title": "Reducing Computational Load", "type": "sec", "content": [
            {"title": "Joint Mirroring", "type": "subsec"},
            {"title": "Temporal Downsampling", "type": "subsec"},
        ]},
        {"title": "Experimental Setup and Results", "type": "sec", "content": [
            {"title": "Loss function and Training Scheme", "type": "subsec"},
            {"title": "Spatio-Temporal Transformer vs. Baselines", "type": "subsec"},
            {"title": "Teacher-Forcing Effect: Loss", "type": "subsec"},
            {"title": "Teacher-Forcing Effect: Temporal Attention", "type": "subsec"},
            {"title": "Embedding Visualization", "type": "subsec"},
        ]},
        {"title": "Experience", "type": "sec", "content": [
            {"title": "Challenges", "type": "subsec"},
            {"title": "Changes in Approach", "type": "subsec"},
            {"title": "Project Success", "type": "subsec"},
        ]},
        {"title": "Work Division", "type": "sec"},
        # Appendix section
        {"title": "Project Code Repository", "type": "sec_appendix"},
        {"title": "Data breakdown", "type": "sec_appendix"},
        {"title": "Masking", "type": "sec_appendix", "content": [
            {"title": "Motion Spatial Attention", "type": "subsec"}
        ]},
        {"title": "Training Scheme", "type": "sec_appendix", "content": [
            {"title": "Training Scheme 1", "type": "subsec"},
            {"title": "Training Scheme 2", "type": "subsec"},
        ]}
    ]

    # 4. Set title and authors
    doc.preamble.append(Command('title', 'PetNet: A Lightweight Framework for Fine-Grained Pet Recognition via Keypoint-Guided Semantic Occlusion\\\\\nNext Move: CS 7643'))

    # Simulate PDF author format
    doc.preamble.append(Command('author', NoEscape(
        r'Haoyi Wang$^*$, Armin Arlert$^*$, Mikasa Ackerman$^*$\\Georgia Institute of Technology\\\{REDACTED\}@gatech.edu')))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    doc.append(NoEscape(r'\maketitle'))

    # 5. Abstract
    doc.append(Command('begin', 'abstract'))
    doc.append(italic(
        "The ability to forecast human motion is useful for a myriad of applications including robotics, self driving cars, and animation... (Copy content from OCR here)"))
    doc.append(Command('end', 'abstract'))

    # 6. Recursive function: Generate PyLaTeX objects based on the structure data
    def build_sections(items, parent_container):
        appendix_started = False

        for item in items:
            # Handle appendix transition
            if item.get("type") == "sec_appendix" and not appendix_started:
                parent_container.append(Command('appendix'))
                appendix_started = True

            if "sec" in item["type"]:  # Section
                with parent_container.create(Section(item["title"])):
                    parent_container.append("Text goes here...")
                    if "content" in item:
                        build_sections(item["content"], parent_container)

            elif item["type"] == "subsec":  # Subsection
                with parent_container.create(Subsection(item["title"])):
                    parent_container.append("Text goes here...")
                    if "content" in item:
                        build_sections(item["content"], parent_container)

            elif item["type"] == "subsubsec":  # Subsubsection
                with parent_container.create(Subsubsection(item["title"])):
                    parent_container.append("Text goes here...")

    # Start building main content
    build_sections(structure, doc)

    # 7. Bibliography placeholder
    doc.append(Command('bibliographystyle', 'plain'))
    doc.append(Command('bibliography', 'references'))  # Assume you have a references.bib

    return doc


if __name__ == '__main__':
    doc = create_paper_template()

    # Generate .tex file
    filename = '../report/motion_prediction_report'
    doc.generate_tex(filename)
    print(f"Successfully generated {filename}.tex")

    # If PDF compiler is installed, uncomment below to generate PDF directly
    try:
        doc.generate_pdf(filename, clean_tex=False)
        print(f"Successfully generated {filename}.pdf")
    except Exception as e:
        print("No local LaTeX compiler detected, only .tex file generated.")