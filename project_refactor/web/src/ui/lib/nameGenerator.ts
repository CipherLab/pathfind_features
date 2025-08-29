const sampleWords = [
    "about", "apple", "because", "people", "thing", "information", "there",
    "which", "their", "would", "important", "family", "example", "another",
    "something", "however", "development", "government", "business", "system",
    "program", "question", "always", "different", "number", "company", "problem",
    "service", "without", "history", "mother", "father", "water", "world",
    "letter", "story", "language", "science", "computer", "technology", "power",
    "control", "knowledge", "learning", "student", "teacher", "country", "national"
];

class WordGenerator {
    private order: number;
    private model: Map<string, string[]>;
    private starters: string[];

    constructor(trainingData: string[], order = 2) {
        if (!trainingData || trainingData.length === 0) {
            throw new Error("Training data cannot be empty.");
        }
        this.order = order;
        this.model = this._buildModel(trainingData);
        this.starters = this._getStarters(trainingData);
    }

    private _buildModel(words: string[]): Map<string, string[]> {
        const model = new Map<string, string[]>();
        const pad = "~".repeat(this.order);

        for (const word of words) {
            const paddedWord = pad + word + ".";
            for (let i = 0; i < paddedWord.length - this.order; i++) {
                const history = paddedWord.substring(i, i + this.order);
                const nextChar = paddedWord[i + this.order];

                if (!model.has(history)) {
                    model.set(history, []);
                }
                model.get(history)!.push(nextChar);
            }
        }
        return model;
    }

    private _getStarters(words: string[]): string[] {
        const starters = words
            .filter(w => w.length >= this.order)
            .map(w => w.substring(0, this.order));
        
        return starters.length > 0 ? starters : ["~".repeat(this.order)];
    }

    public generateWord(minLength = 4, maxLength = 10): string {
        while (true) {
            const start = this.starters[Math.floor(Math.random() * this.starters.length)];
            let result = start;
            
            for (let i = 0; i < maxLength; i++) {
                const history = result.substring(result.length - this.order);
                if (!this.model.has(history)) {
                    break;
                }
                const nextChars = this.model.get(history)!;
                const nextChar = nextChars[Math.floor(Math.random() * nextChars.length)];

                if (nextChar === ".") {
                    break;
                }
                result += nextChar;
            }

            const finalWord = result.replace(/~/g, "");
            if (finalWord.length >= minLength && finalWord.length <= maxLength) {
                return finalWord.charAt(0).toUpperCase() + finalWord.slice(1);
            }
        }
    }
}

const generator = new WordGenerator(sampleWords, 2);

export function generateExperimentName(): string {
    const word1 = generator.generateWord(4, 7);
    const word2 = generator.generateWord(4, 7);
    const num = Math.floor(Math.random() * 1000);
    return `${word1}${word2}${num}`;
}