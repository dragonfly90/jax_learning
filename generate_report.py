import os
import re
import nbformat
from nbconvert import HTMLExporter

def extract_body_content(html):
    """Extract content between <body> and </body>, stripping the full HTML wrapper."""
    match = re.search(r'<body[^>]*>(.*)</body>', html, re.DOTALL)
    if match:
        return match.group(1)
    return html

def generate_report():
    notebook_dir = 'notebooks'
    notebook_files = sorted([f for f in os.listdir(notebook_dir) if f.endswith('.ipynb')])

    html_exporter = HTMLExporter()
    html_exporter.template_name = 'lab'
    html_exporter.exclude_anchor_links = True

    # Collect CSS from the first notebook export to include once
    collected_styles = []
    all_lessons = []

    for i, filename in enumerate(notebook_files):
        path = os.path.join(notebook_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            (body, resources) = html_exporter.from_notebook_node(nb)

            # Extract styles from the first notebook only (they're all the same)
            if i == 0:
                style_matches = re.findall(r'<style[^>]*>(.*?)</style>', body, re.DOTALL)
                collected_styles = style_matches

            # Extract only the body content, removing full HTML document wrapper
            content = extract_body_content(body)

            title = filename.replace('.ipynb', '').split('_', 1)[1].replace('_', ' ').title()

            all_lessons.append({
                'id': filename.split('_')[0],
                'title': title,
                'content': content
            })

    # Build the notebook styles block (from nbconvert, included once)
    nb_styles = '\n'.join(f'<style>{s}</style>' for s in collected_styles)

    # Prepare the final HTML template
    template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JAX Learning: Interactive Tutorial</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    {nb_styles}
    <style>
        :root {{
            --bg-dark: #0a0a0c;
            --bg-card: rgba(255, 255, 255, 0.05);
            --accent-primary: #ff9800; /* JAX Orangeish */
            --accent-secondary: #2196f3; /* JAX Blueish */
            --text-main: #e0e0e0;
            --text-muted: #a0a0a0;
            --glass-border: rgba(255, 255, 255, 0.1);
            --font-main: 'Inter', sans-serif;
            --font-mono: 'Fira Code', monospace;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: var(--font-main);
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-dark);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--accent-primary);
            border-radius: 4px;
        }}

        /* Header */
        header {{
            position: fixed;
            top: 0;
            width: 100%;
            padding: 1.5rem 3rem;
            background: rgba(10, 10, 12, 0.8);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--glass-border);
        }}

        .logo {{
            font-weight: 800;
            font-size: 1.5rem;
            background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }}

        /* Layout */
        .container {{
            display: flex;
            margin-top: 80px;
            min-height: calc(100vh - 80px);
        }}

        /* Sidebar Nav */
        nav {{
            width: 300px;
            height: calc(100vh - 80px);
            position: fixed;
            padding: 2rem;
            overflow-y: auto;
            border-right: 1px solid var(--glass-border);
            background: rgba(255, 255, 255, 0.02);
        }}

        .nav-item {{
            display: block;
            padding: 0.8rem 1rem;
            color: var(--text-muted);
            text-decoration: none;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
            font-size: 0.9rem;
            border: 1px solid transparent;
        }}

        .nav-item:hover, .nav-item.active {{
            background: var(--bg-card);
            color: white;
            border-color: var(--glass-border);
            transform: translateX(5px);
        }}

        .nav-item.active {{
            border-left: 4px solid var(--accent-primary);
        }}

        /* Main Content */
        main {{
            margin-left: 300px;
            flex: 1;
            padding: 3rem;
            background-color: var(--bg-dark);
        }}

        main > * {{
            max-width: 1000px;
        }}

        section {{
            margin-bottom: 6rem;
            animation: fadeIn 0.8s ease-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Jupyter Lab / Notebook Overrides */
        .jp-Notebook {{
            background-color: transparent !important;
            padding: 0 !important;
        }}

        .jp-Cell {{
            margin-bottom: 2rem !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            color: var(--text-main) !important;
        }}

        .jp-InputArea-editor {{
            background: rgba(0,0,0,0.3) !important;
            border: none !important;
        }}

        .jp-InputArea-prompt, .jp-OutputArea-prompt {{
            display: none !important; /* Hide [In/Out] prompts for cleaner look */
        }}

        .jp-RenderedText pre {{
            color: #8bc34a !important; /* Accent color for output */
            background: none !important;
        }}

        .jp-MarkdownCell {{
            background: transparent !important;
            color: var(--text-main) !important;
            margin-bottom: 1.5rem !important;
        }}
        
        .jp-MarkdownCell h1, .jp-MarkdownCell h2, .jp-MarkdownCell h3 {{
            color: var(--accent-primary) !important;
            border: none !important;
            margin-top: 2rem !important;
        }}

        .jp-OutputArea-output {{
            background: rgba(0,0,0,0.2) !important;
            padding: 1rem !important;
        }}

        /* Code Mirror Theme Overrides */
        .cm-s-jupyter span.cm-keyword {{ color: #ff7b72 !important; }}
        .cm-s-jupyter span.cm-def {{ color: #d2a8ff !important; }}
        .cm-s-jupyter span.cm-variable {{ color: #ffa657 !important; }}
        .cm-s-jupyter span.cm-string {{ color: #a5d6ff !important; }}
        .cm-s-jupyter span.cm-comment {{ color: #8b949e !important; }}

        h1, h2, h3 {{
            margin-bottom: 1.5rem;
            line-height: 1.2;
        }}

        h1 {{
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -2px;
            margin-bottom: 2rem;
            background: linear-gradient(to right, #fff, #888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        /* Hero Section */
        .hero {{
            text-align: center;
            padding: 6rem 0;
            background: radial-gradient(circle at center, rgba(255, 152, 0, 0.1) 0%, transparent 70%);
        }}

        .hero h1 {{
            margin-bottom: 1rem;
        }}
        
        .hero p {{
            font-size: 1.2rem;
            color: var(--text-muted);
            max-width: 600px;
            margin: 0 auto;
        }}

        /* Interactive bits */
        .copy-btn {{
            position: absolute;
            right: 1rem;
            top: 1rem;
            background: var(--glass-border);
            border: none;
            color: white;
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.7rem;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .jp-Cell:hover .copy-btn {{
            opacity: 1;
        }}

        /* Force dark theme on all notebook elements */
        .jp-Notebook, .jp-Cell, .jp-InputArea, .jp-OutputArea,
        .jp-RenderedHTMLCommon, .jp-RenderedMarkdown,
        div.jp-Cell-inputWrapper, div.jp-Cell-outputWrapper {{
            background-color: transparent !important;
            color: var(--text-main) !important;
        }}

        .highlight, .highlight pre {{
            background: rgba(0,0,0,0.3) !important;
            color: var(--text-main) !important;
        }}

        .jp-RenderedHTMLCommon table {{
            color: var(--text-main) !important;
        }}

        .jp-RenderedHTMLCommon a {{
            color: var(--accent-secondary) !important;
        }}

        .jp-RenderedHTMLCommon code {{
            background: rgba(255,255,255,0.1) !important;
            color: var(--accent-primary) !important;
            padding: 0.15em 0.4em !important;
            border-radius: 4px !important;
        }}

        .jp-RenderedHTMLCommon pre code {{
            background: transparent !important;
            color: inherit !important;
            padding: 0 !important;
        }}

    </style>
</head>
<body>
    <header>
        <div class="logo">JAX LEARNING</div>
        <div>
            <a href="https://github.com/dragonfly90/jax_learning" style="color: white; text-decoration: none; font-size: 0.9rem;">View Repo</a>
        </div>
    </header>

    <div class="container">
        <nav id="sidebar">
            <div style="margin-bottom: 2rem; padding: 0 1rem;">
                <p style="font-weight: 600; font-size: 0.7rem; color: var(--accent-primary); letter-spacing: 2px; text-transform: uppercase;">Tutorial Index</p>
            </div>
            {"".join([f'<a href="#lesson-{l["id"]}" class="nav-item"><b>{l["id"]}</b> {l["title"]}</a>' for l in all_lessons])}
        </nav>

        <main>
            <div class="hero">
                <h1>High-Performance JAX</h1>
                <p>A comprehensive visual guide from basics to advanced distributed training on TPU.</p>
            </div>

            {"".join([f'<section id="lesson-{l["id"]}"> {l["content"]} </section>' for l in all_lessons])}
            
            <footer style="margin-top: 10rem; padding-bottom: 5rem; border-top: 1px solid var(--glass-border); padding-top: 3rem; text-align: center; color: var(--text-muted);">
                <p>Generated with ✨ Antigravity Agent • 2026</p>
            </footer>
        </main>
    </div>

    <script>
        // Smooth scroll for nav items
        document.querySelectorAll('.nav-item').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                window.scrollTo({{
                    top: target.offsetTop - 100,
                    behavior: 'smooth'
                }});
            }});
        }});

        // Highlight active nav item on scroll
        const sections = document.querySelectorAll('section');
        const navItems = document.querySelectorAll('.nav-item');

        window.addEventListener('scroll', () => {{
            let current = "";
            sections.forEach(section => {{
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (pageYOffset >= (sectionTop - 150)) {{
                    current = section.getAttribute('id');
                }}
            }});

            navItems.forEach(item => {{
                item.classList.remove('active');
                if (item.getAttribute('href') === `#${{current}}`) {{
                    item.classList.add('active');
                }}
            }});
        }});
    </script>
</body>
</html>
"""

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(template)

if __name__ == "__main__":
    generate_report()
