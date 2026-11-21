using System.Reflection;
using Couchbase;
using Couchbase.VectorData;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using OpenAI;

namespace CouchbaseVectorSearchDemo;

/// <summary>
/// Couchbase Vector Search Demo
///
/// This example demonstrates how to use CouchbaseQueryCollection with Hyperscale index for vector search.
/// </summary>
public abstract class Program
{
    private const string IndexName = "hyperscale_glossary_index";

    public static async Task Main(string[] args)
    {
        Console.WriteLine("Couchbase Hyperscale Vector Search Demo");
        Console.WriteLine("====================================");

        ICluster? cluster = null;

        try
        {
            // Setup configuration and services
            var configuration = BuildConfiguration();
            var embeddingGenerator = CreateEmbeddingGenerator(configuration);
            var couchbaseSettings = CouchbaseSettings.FromConfiguration(configuration);

            cluster = await Cluster.ConnectAsync(couchbaseSettings.ConnectionString, couchbaseSettings.Username, couchbaseSettings.Password);
            var collection = await GetGlossaryCollectionAsync(cluster, couchbaseSettings);

            // Step 1: Ingest data into Couchbase vector store
            await IngestDataIntoCouchbaseVectorStoreAsync(collection, embeddingGenerator);

            // Step 2: Create Hyperscale index after ingestion
            await CreateHyperscaleIndexAsync(collection);

            // Step 3: Perform vector search
            await SearchCouchbaseVectorStoreAsync(collection, embeddingGenerator);

            // Step 4: Perform filtered vector search
            await SearchCouchbaseVectorStoreWithFilteringAsync(collection, embeddingGenerator);

            Console.WriteLine("\n Demo completed successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n Full error: {ex}");
        }
        finally
        {
            if (cluster is not null)
            {
                await cluster.DisposeAsync();
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }

    /// <summary>
    /// Couchbase connection and keyspace settings read from configuration.
    /// </summary>
    private sealed record CouchbaseSettings(
        string ConnectionString,
        string Username,
        string Password,
        string BucketName,
        string ScopeName,
        string CollectionName)
    {
        public static CouchbaseSettings FromConfiguration(IConfiguration configuration)
        {
            var connectionString = configuration["Couchbase:ConnectionString"];
            var username = configuration["Couchbase:Username"];
            var password = configuration["Couchbase:Password"];
            var bucketName = configuration["Couchbase:BucketName"];
            var scopeName = configuration["Couchbase:ScopeName"];
            var collectionName = configuration["Couchbase:CollectionName"];

            if (string.IsNullOrEmpty(connectionString) || string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password) ||
                string.IsNullOrEmpty(bucketName) || string.IsNullOrEmpty(scopeName) || string.IsNullOrEmpty(collectionName))
            {
                throw new InvalidOperationException(
                    "Couchbase configuration missing. Please set ConnectionString, Username, Password, BucketName, ScopeName, CollectionName in appsettings.json or user secrets.");
            }

            return new CouchbaseSettings(connectionString, username, password, bucketName, scopeName, collectionName);
        }
    }

    /// <summary>
    /// Opens the configured scope and returns the vector store collection, created once and reused across all steps.
    /// </summary>
    private static async Task<CouchbaseQueryCollection<string, Glossary>> GetGlossaryCollectionAsync(
        ICluster cluster, CouchbaseSettings settings)
    {
        var bucket = await cluster.BucketAsync(settings.BucketName);
        var scope = await bucket.ScopeAsync(settings.ScopeName);

        var collectionOptions = new CouchbaseQueryCollectionOptions
        {
            IndexName = IndexName,
            SimilarityMetric = "cosine"
        };

        var vectorStore = new CouchbaseVectorStore(scope);
        return (CouchbaseQueryCollection<string, Glossary>)vectorStore.GetCollection<string, Glossary>(settings.CollectionName, collectionOptions);
    }

    /// <summary>
    /// Build configuration from appsettings.json, user secrets, and environment variables.
    /// </summary>
    private static IConfiguration BuildConfiguration()
    {
        return new ConfigurationBuilder()
            .AddJsonFile("appsettings.Development.json", optional: true, reloadOnChange: true)
            .AddEnvironmentVariables()
            .AddUserSecrets(Assembly.GetExecutingAssembly())
            .Build();
    }

    /// <summary>
    /// Create the OpenAI embedding generator.
    /// </summary>
    private static IEmbeddingGenerator<string, Embedding<float>> CreateEmbeddingGenerator(IConfiguration configuration)
    {
        var openAiApiKey = configuration["OpenAI:ApiKey"];
        var openAiModel = configuration["OpenAI:EmbeddingModel"] ?? "text-embedding-3-small";

        if (string.IsNullOrEmpty(openAiApiKey))
        {
            throw new InvalidOperationException(
                "OpenAI API key missing. Please set OpenAI:ApiKey in appsettings.json or user secrets. ");
        }

        Console.WriteLine($"Using OpenAI model: {openAiModel}");

        return new OpenAIClient(openAiApiKey)
            .GetEmbeddingClient(openAiModel)
            .AsIEmbeddingGenerator();
    }

    /// <summary>
    /// Ingest data into Couchbase vector store.
    /// </summary>
    /// <remarks>
    /// Uses only the Semantic Kernel <see cref="VectorStoreCollection{TKey,TRecord}"/> abstraction, so it works unchanged
    /// with any Couchbase collection type (Hyperscale, Composite or Search vector index).
    /// </remarks>
    private static async Task IngestDataIntoCouchbaseVectorStoreAsync(
        VectorStoreCollection<string, Glossary> collection,
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        Console.WriteLine("Step 1: Ingesting data into Couchbase vector store...");

        var glossaryEntries = CreateGlossaryEntries().ToList();

        // Generate all embeddings in a single batched request.
        var embeddings = await embeddingGenerator.GenerateAsync(glossaryEntries.Select(entry => entry.Definition));
        for (var i = 0; i < glossaryEntries.Count; i++)
        {
            glossaryEntries[i].DefinitionEmbedding = embeddings[i].Vector;
        }

        await collection.UpsertAsync(glossaryEntries);

        Console.WriteLine("Data ingestion completed");
    }

    /// <summary>
    /// Create the Hyperscale vector index after documents are inserted.
    /// </summary>
    private static async Task CreateHyperscaleIndexAsync(CouchbaseQueryCollection<string, Glossary> collection)
    {
        Console.WriteLine("\nStep 2: Creating Hyperscale vector index...");

        // The connector builds and runs the CREATE VECTOR INDEX statement, deriving the dimensions,
        // similarity metric and INCLUDE fields from the record model and collection options.
        try
        {
            Console.WriteLine("Executing Hyperscale index creation query...");
            await collection.EnsureVectorIndexExistsAsync();
            Console.WriteLine($"Hyperscale vector index '{IndexName}' is ready.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to create Hyperscale vector index: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Perform basic vector search.
    /// </summary>
    private static async Task SearchCouchbaseVectorStoreAsync(
        VectorStoreCollection<string, Glossary> collection,
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        Console.WriteLine("\nStep 3: Performing vector search...");

        var searchVector = (await embeddingGenerator.GenerateAsync("What is an Application Programming Interface?")).Vector;

        var searchResultItem = (await collection.SearchAsync(searchVector, top: 1).ToListAsync()).FirstOrDefault();

        Console.WriteLine($"   Found: {searchResultItem?.Record.Term}");
        Console.WriteLine($"   Definition: {searchResultItem?.Record.Definition}");
        Console.WriteLine($"   Score: {searchResultItem?.Score:F4}");
    }

    /// <summary>
    /// Perform filtered vector search.
    /// </summary>
    private static async Task SearchCouchbaseVectorStoreWithFilteringAsync(
        VectorStoreCollection<string, Glossary> collection,
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        Console.WriteLine("\nStep 4: Performing filtered vector search...");

        var searchVector = (await embeddingGenerator.GenerateAsync("How do I provide additional context to an LLM?")).Vector;

        // Search the store with a filter and get the single most relevant result
        var searchResultItems = await collection.SearchAsync(
            searchVector,
            top: 1,
            new VectorSearchOptions<Glossary>
            {
                Filter = g => g.Category == "AI"
            }).ToListAsync();

        if (searchResultItems.Count != 0)
        {
            var result = searchResultItems.FirstOrDefault();
            Console.WriteLine($"   Found (AI category only): {result?.Record.Term}");
            Console.WriteLine($"   Definition: {result?.Record.Definition}");
            Console.WriteLine($"   Score: {result?.Score:F4}");
        }
        else
        {
            Console.WriteLine("No results found with AI category filter");
        }
    }

    /// <summary>
    /// Create some sample glossary entries.
    /// </summary>
    /// <returns>A list of sample glossary entries.</returns>
    private static IEnumerable<Glossary> CreateGlossaryEntries()
    {
        yield return new Glossary
        {
            Key = "1",
            Category = "Software",
            Term = "API",
            Definition = "Application Programming Interface. A set of rules and specifications that allow software components to communicate and exchange data."
        };

        yield return new Glossary
        {
            Key = "2",
            Category = "Software",
            Term = "SDK",
            Definition = "Software development kit. A set of libraries and tools that allow software developers to build software more easily."
        };

        yield return new Glossary
        {
            Key = "3",
            Category = "SK",
            Term = "Connectors",
            Definition = "Semantic Kernel Connectors allow software developers to integrate with various services providing AI capabilities, including LLM, AudioToText, TextToAudio, Embedding generation, etc."
        };

        yield return new Glossary
        {
            Key = "4",
            Category = "SK",
            Term = "Semantic Kernel",
            Definition = "Semantic Kernel is a set of libraries that allow software developers to more easily develop applications that make use of AI experiences."
        };

        yield return new Glossary
        {
            Key = "5",
            Category = "AI",
            Term = "RAG",
            Definition = "Retrieval Augmented Generation - a term that refers to the process of retrieving additional data to provide as context to an LLM to use when generating a response (completion) to a user's question (prompt)."
        };

        yield return new Glossary
        {
            Key = "6",
            Category = "AI",
            Term = "LLM",
            Definition = "Large language model. A type of artificial intelligence algorithm that is designed to understand and generate human language."
        };
    }
}
