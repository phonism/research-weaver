# Research Weaver - GitHub Pages

This directory contains the GitHub Pages website for Research Weaver.

## 🌐 Live Website

Visit the live website at: `https://phonism.github.io/research-weaver/`

## 📁 Directory Structure

```
docs/
├── index.html              # Main landing page
├── documentation.html      # Detailed documentation
├── demos/                  # Demo files
│   ├── ResearchWeaver-demo1.html
│   └── ResearchWeaver-demo1_files/
└── README.md              # This file
```

## 🚀 Setting Up GitHub Pages

To enable GitHub Pages for this repository:

1. **Go to Repository Settings**
   - Navigate to your GitHub repository
   - Click on "Settings" tab

2. **Configure Pages**
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/docs" folder
   - Click "Save"

3. **Access Your Site**
   - Your site will be available at: `https://phonism.github.io/research-weaver/`
   - It may take a few minutes for changes to deploy

## 📝 Customization

### Update Repository Links

Replace `phonism` in the following files with your actual GitHub username if needed:

- `index.html`: Update all GitHub links
- `documentation.html`: Update repository references

### Add Your Demo

To add your own demo:

1. Export your Streamlit app as HTML
2. Save it in the `demos/` directory
3. Update the iframe source in `index.html`

### Customize Content

- **Hero Section**: Update title, description, and call-to-action buttons
- **Features**: Modify feature cards to highlight your specific capabilities
- **How It Works**: Adjust the process steps to match your implementation
- **Contact Info**: Add your contact information in the footer

## 🎨 Design Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, professional design with smooth animations
- **Interactive Demo**: Embedded Streamlit demo for live testing
- **Documentation**: Comprehensive setup and usage guide
- **SEO Optimized**: Meta tags and structured content for search engines

## 🔧 Technical Details

- **Pure HTML/CSS/JS**: No build process required
- **CDN Resources**: Uses Font Awesome, Google Fonts, and Prism.js
- **Performance**: Optimized for fast loading
- **Accessibility**: Semantic HTML and proper ARIA labels

## 📱 Mobile Optimization

The website is fully responsive and optimized for:
- Desktop (1200px+)
- Tablet (768px - 1199px)
- Mobile (< 768px)

## 🎯 Call-to-Action Flow

1. **Landing**: Hero section with clear value proposition
2. **Features**: Showcase key capabilities
3. **Demo**: Interactive demonstration
4. **Documentation**: Detailed setup guide
5. **GitHub**: Direct link to repository

## 🚀 Deployment Notes

- Changes to files in `/docs` automatically deploy to GitHub Pages
- Allow 5-10 minutes for changes to appear
- Check the "Actions" tab for deployment status
- Custom domains can be configured in repository settings

## 📊 Analytics (Optional)

To add Google Analytics, insert your tracking code before the closing `</head>` tag in both HTML files:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
```

## 🤝 Contributing

To contribute to the website:

1. Fork the repository
2. Make changes to files in `/docs`
3. Test locally by opening HTML files in browser
4. Submit a pull request

## 📞 Support

For website-related issues:
- Open an issue in the GitHub repository
- Include browser information and screenshots
- Describe the expected vs actual behavior

---

**Happy researching with Research Weaver! 🔬✨**