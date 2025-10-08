package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/chromedp/chromedp"
	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

// Constants for file paths and directories.
const (
	urlsFile       = "urls.txt"
	baseResumeFile = "resume.md"
	outputDir      = "output"
	savedPagesDir  = "saved_pages"
)

// JobDetails holds scraped job information.
type JobDetails struct {
	SourceFile  string
	Company     string
	Title       string
	Description string
}

// Scraper manages the browser instance and its context.
type Scraper struct {
	ctx    context.Context
	cancel context.CancelFunc
}

// NewScraper initializes a new browser instance.
func NewScraper() (*Scraper, error) {
	ctx, cancel := chromedp.NewExecAllocator(context.Background(), chromedp.DefaultExecAllocatorOptions[:]...)
	return &Scraper{ctx: ctx, cancel: cancel}, nil
}

// Close terminates the browser instance.
func (s *Scraper) Close() {
	s.cancel()
}

// scrapeJobDetails automates LinkedIn login, page navigation, and content extraction.
func (s *Scraper) scrapeJobDetails(url, email, password string) (*JobDetails, error) {
	fmt.Printf("Starting process for job URL: %s\n", url)

	// Create a new browser tab context for this specific job.
	taskCtx, cancel := chromedp.NewContext(s.ctx)
	defer cancel()

	var pageSource string
	fmt.Println("Logging into LinkedIn...")

	// Login sequence.
	if err := chromedp.Run(taskCtx,
		chromedp.Navigate("https://www.linkedin.com/login"),
		chromedp.WaitVisible("#username"),
		chromedp.SendKeys("#username", email),
		chromedp.SendKeys("#password", password),
		chromedp.Click("button[type='submit']"),
		chromedp.WaitVisible("#global-nav", chromedp.ByQuery),
	); err != nil {
		return nil, fmt.Errorf("login failed (check credentials, CAPTCHA, or 2FA): %w", err)
	}
	fmt.Println("Login successful.")

	// Navigate to job page and extract HTML.
	fmt.Println("Navigating to job page...")
	if err := chromedp.Run(taskCtx,
		chromedp.Navigate(url),
		chromedp.WaitVisible("#job-details", chromedp.ByID),
		chromedp.OuterHTML("html", &pageSource),
	); err != nil {
		return nil, fmt.Errorf("failed to load job page details: %w", err)
	}
	fmt.Println("Job page content loaded.")

	// Save the page source to a file.
	filePath, err := savePageSource(url, pageSource)
	if err != nil {
		return nil, err
	}
	fmt.Printf("✅ Page successfully saved as: %s\n", filePath)

	// Parse the saved HTML.
	return parseJobDetails(filePath, pageSource)
}

// savePageSource saves the HTML content to a file.
func savePageSource(url, source string) (string, error) {
	re := regexp.MustCompile(`/view/(\d+)/`)
	matches := re.FindStringSubmatch(url)
	filename := "job_details.html"
	if len(matches) > 1 {
		filename = fmt.Sprintf("job_%s.html", matches[1])
	}

	if err := os.MkdirAll(savedPagesDir, 0755); err != nil {
		return "", fmt.Errorf("could not create save directory: %w", err)
	}
	filePath := filepath.Join(savedPagesDir, filename)
	if err := os.WriteFile(filePath, []byte(source), 0644); err != nil {
		return "", fmt.Errorf("could not save page source: %w", err)
	}
	return filePath, nil
}

// parseJobDetails extracts job information from the HTML source.
func parseJobDetails(filePath, source string) (*JobDetails, error) {
	fmt.Printf("Parsing data from the saved file: %s\n", filePath)
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(source))
	if err != nil {
		return nil, fmt.Errorf("could not parse saved HTML: %w", err)
	}

	fullTitle := doc.Find("title").First().Text()
	parts := strings.Split(fullTitle, "|")
	jobTitle, company := "N/A", "N/A"
	if len(parts) >= 2 {
		jobTitle = strings.TrimSpace(parts[0])
		company = strings.TrimSpace(parts[1])
	}

	description := doc.Find("#job-details").First().Text()
	description = strings.TrimSpace(regexp.MustCompile(`\s+`).ReplaceAllString(description, " "))
	if description == "" {
		description = "Job Description not found."
	} else {
		fmt.Println("✅ Successfully extracted job details.")
	}

	return &JobDetails{
		SourceFile:  filePath,
		Company:     company,
		Title:       jobTitle,
		Description: description,
	}, nil
}

// generateContent calls the Gemini API with a given prompt.
func generateContent(ctx context.Context, model *genai.GenerativeModel, prompt string) (string, error) {
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("gemini API call failed: %w", err)
	}
	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("gemini returned no content")
	}

	var builder strings.Builder
	for _, cand := range resp.Candidates {
		for _, part := range cand.Content.Parts {
			if txt, ok := part.(genai.Text); ok {
				builder.WriteString(string(txt))
			}
		}
	}
	return builder.String(), nil
}

// tailorResume generates a tailored resume using the Gemini API.
func tailorResume(ctx context.Context, model *genai.GenerativeModel, baseResume string, details *JobDetails) (string, error) {
	fmt.Printf("Tailoring resume for %s at %s...\n", details.Title, details.Company)
	prompt := fmt.Sprintf(`
	You are an expert career coach. Tailor the following base resume for the given job description.

	**Base Resume:**
	---
	%s
	---
	**Job Description:**
	---
	Company: %s
	Job Title: %s
	Description: %s
	---
	**Instructions:**
	1. Rewrite the summary to target the job's key requirements.
	2. Rephrase experience bullet points to highlight relevant skills and quantify achievements.
	3. Prioritize skills in the 'Skills' section that match the job description.
	4. Maintain the original Markdown format and a professional tone.
	5. Only rephrase and reorder existing information; do not add new content.
	
	Return only the tailored resume in Markdown.`, baseResume, details.Company, details.Title, details.Description)

	return generateContent(ctx, model, prompt)
}

// generateCoverLetter creates a cover letter using the Gemini API.
func generateCoverLetter(ctx context.Context, model *genai.GenerativeModel, tailoredResume string, details *JobDetails) (string, error) {
	fmt.Printf("Generating cover letter for %s at %s...\n", details.Title, details.Company)
	prompt := fmt.Sprintf(`
	You are a professional cover letter writer. Write a compelling and concise cover letter based on the provided resume and job description.

	**Tailored Resume:**
	---
	%s
	---
	**Job Description:**
	---
	Company: %s
	Job Title: %s
	Description: %s
	---
	**Instructions:**
	1. Structure the letter with an introduction, body, and conclusion.
	2. Highlight 2-3 key experiences from the resume that align with the job's requirements.
	3. Keep the tone professional and confident, around 3-4 paragraphs.
	4. Address it to the "Hiring Manager".
	5. If the location is in the Netherlands, mention readiness to relocate and visa requirements.
	
	Return only the cover letter text.`, tailoredResume, details.Company, details.Title, details.Description)

	return generateContent(ctx, model, prompt)
}

// sanitizeFilename removes characters that are invalid in filenames.
func sanitizeFilename(name string) string {
	return regexp.MustCompile(`[<>:"/\\|?*]`).ReplaceAllString(name, "")
}

// saveOutput writes content to a file in the output directory.
func saveOutput(filename, content string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}
	path := filepath.Join(outputDir, filename)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return err
	}
	fmt.Printf("Successfully saved: %s\n", filename)
	return nil
}

// AppConfig holds the application's configuration.
type AppConfig struct {
	LinkedInEmail    string
	LinkedInPassword string
	GeminiAPIKey     string
}

// loadConfig loads configuration from environment variables.
func loadConfig() (*AppConfig, error) {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on system environment variables.")
	}
	config := &AppConfig{
		LinkedInEmail:    os.Getenv("LINKEDIN_EMAIL"),
		LinkedInPassword: os.Getenv("LINKEDIN_PASSWORD"),
		GeminiAPIKey:     os.Getenv("GEMINI_API_KEY"),
	}
	if config.LinkedInEmail == "" || config.LinkedInPassword == "" || config.GeminiAPIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY, LINKEDIN_EMAIL, and LINKEDIN_PASSWORD must be set")
	}
	return config, nil
}

// handleJobURL processes a single job application URL.
func handleJobURL(scraper *Scraper, geminiModel *genai.GenerativeModel, config *AppConfig, baseResume, url string) {
	jobDetails, err := scraper.scrapeJobDetails(url, config.LinkedInEmail, config.LinkedInPassword)
	if err != nil {
		log.Printf("Skipping URL %s due to scraping error: %v", url, err)
		return
	}

	ctx := context.Background()
	tailoredResume, err := tailorResume(ctx, geminiModel, baseResume, jobDetails)
	if err != nil {
		log.Printf("Failed to tailor resume for %s: %v", url, err)
		return
	}

	coverLetter, err := generateCoverLetter(ctx, geminiModel, tailoredResume, jobDetails)
	if err != nil {
		log.Printf("Failed to generate cover letter for %s: %v", url, err)
		return
	}

	sanitizedCompany := sanitizeFilename(jobDetails.Company)
	sanitizedTitle := sanitizeFilename(jobDetails.Title)
	resumeFilename := fmt.Sprintf("%s - %s - Resume.md", sanitizedCompany, sanitizedTitle)
	coverLetterFilename := fmt.Sprintf("%s - %s - Cover Letter.md", sanitizedCompany, sanitizedTitle)

	if err := saveOutput(resumeFilename, tailoredResume); err != nil {
		log.Printf("Error saving resume: %v", err)
	}
	if err := saveOutput(coverLetterFilename, coverLetter); err != nil {
		log.Printf("Error saving cover letter: %v", err)
	}
}

func main() {
	var start = time.Now()
	config, err := loadConfig()
	if err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(config.GeminiAPIKey))
	if err != nil {
		log.Fatalf("Failed to create Gemini client: %v", err)
	}
	defer client.Close()
	geminiModel := client.GenerativeModel("gemini-2.5-flash")

	baseResumeBytes, err := os.ReadFile(baseResumeFile)
	if err != nil {
		log.Fatalf("Failed to read base resume file (%s): %v", baseResumeFile, err)
	}
	baseResumeContent := string(baseResumeBytes)

	file, err := os.Open(urlsFile)
	if err != nil {
		log.Fatalf("Failed to open URLs file (%s): %v", urlsFile, err)
	}
	defer file.Close()

	scraper, err := NewScraper()
	if err != nil {
		log.Fatalf("Failed to initialize scraper: %v", err)
	}
	defer scraper.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		url := strings.TrimSpace(scanner.Text())
		if url == "" {
			continue
		}
		handleJobURL(scraper, geminiModel, config, baseResumeContent, url)
		fmt.Println(strings.Repeat("-", 40))
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading from URLs file: %v", err)
	}
	fmt.Printf("Total time taken: %v\n", time.Since(start))
}
