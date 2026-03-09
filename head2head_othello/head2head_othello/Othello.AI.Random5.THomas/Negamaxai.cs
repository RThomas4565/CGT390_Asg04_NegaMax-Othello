


using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Othello.Contract;

namespace Othello.AI.Random;

/// <summary>
/// NegaMax AI - An intelligent Othello player using the NegaMax algorithm with Alpha-Beta pruning.
/// 
/// ALGORITHM OVERVIEW:
/// - NegaMax: A recursive minimax variant that uses a single function to evaluate both player perspectives
/// - Alpha-Beta Pruning: An optimization that eliminates branches guaranteed to be irrelevant to the final decision
/// - Search Depth: Explores moves 8 levels deep into the game tree
/// - Evaluation: Scores board positions using positional weighting (corners are most valuable)
/// 
/// WHY THIS APPROACH:
/// Random AI picks moves with no foresigh NegaMax plans ahead by RUSSELL
/// 1. Simulating possible moves several turns into the future
/// 2. Evaluating the resulting board positions
/// 3. Backtracking scores to find the objectively best move
/// 4. Pruning the search to avoid wasting time on hopeless branches
/// </summary>
public class NegaMaxAI : IOthelloAI
{
    // ============================================
    // CONFIGURATION CONSTANTS
    // ============================================

    /// <summary>
    /// How many moves ahead to search. Depth 8 = search ~8 moves into the future.
    /// Higher depth = stronger AI but slower execution.
    /// Depth 8 is a sweet spot: strong enough to beat random players consistently,
    /// fast enough to respond in reasonable time.
    /// </summary>
    private const int MaxDepth = 6;

    /// <summary>
    /// BOARD POSITION VALUES - These define how valuable different squares are.
    /// The AI uses these weights when evaluating board positions.
    /// 
    /// DESIGN RATIONALE:
    /// In Othello, not all squares are equally valuable:
    /// - Corners (400): Extremely valuable. Once you own a corner, opponent can't flip it.
    ///   This is the most important strategic goal in Othello
    /// - Squares next to corners -100  They allow opponent to take corners
    ///   The AI heavily avoids putting pieces here early game
    /// - Regular edges 20, 10, -5 Moderately valuable. Hard to flip but not as secure as corners.
    /// - Center squares 1-5 Neutral territory Easy to flip least important
    /// 
    /// This weighting strategy is based on standard Othello strategy and competitive play analysis.
    /// </summary>
    private const int CornerValue = 100;
    private const int EdgeValue = 10;
    private const int NormalValue = 1;

    public string Name => "NegaMax AI";

    // ============================================
    // MAIN ENTRY POINT
    // ============================================

    /// <summary>
    /// Called by the game engine whenever it's this AI's turn.
    /// This is the function that gets invoked each move.
    /// 
    /// WORKFLOW:
    /// 1. Get all legal moves available
    /// 2. If no legal moves, return null (pass turn)
    /// 3. Use NegaMax to evaluate each legal move
    /// 4. Return the move with the highest score
    /// 5. Use Alpha-Beta pruning to skip evaluating bad branches
    /// 
    /// PARAMETERS:
    /// - board: Current game state (8x8 grid of pieces)
    /// - yourColor: Whether this AI is Black or White
    /// - ct: Cancellation token (allows game to interrupt if needed)
    /// 
    /// RETURNS:
    /// - A Move (row, column) representing the best move, or null if no moves available
    /// </summary>
    public async Task<Move?> GetMoveAsync(BoardState board, DiscColor yourColor, CancellationToken ct)
    {
        // Small delay to make the game feel more natural like a player palying the game
        await Task.Delay(50, ct);

        // STEP 1: Find all legal moves
        var validMoves = GetValidMoves(board, yourColor);

        // STEP 2: If no moves available, this player must pass
        if (validMoves.Count == 0) return null;

        // STEP 3: Initialize variables for tracking the best move
        var bestMove = validMoves[0];  // Default to first move
        int bestScore = int.MinValue;   // Track best score found so far
        int alpha = int.MinValue;       // Alpha: best score WE can guarantee
        int beta = int.MaxValue;        // Beta: best score OPPONENT can guarantee

        // STEP 4: Evaluate each legal move using NegaMax
        foreach (var move in validMoves)
        {
            // Create a copy of the board to test this move without modifying the real one
            var newBoard = board.Clone();
            ApplyMove(newBoard, move, yourColor);

            // After we make this move, it becomes the opponent's turn
            DiscColor opponent = yourColor == DiscColor.Black ? DiscColor.White : DiscColor.Black;

            // CRITICAL: Use NegaMax to find the score of this move
            // The score is NEGATED because NegaMax returns scores from the current player's perspective
            // We're evaluating from OUR perspective, so we negate the opponent's perspective
            int score = -NegaMax(newBoard, MaxDepth - 1, opponent, -beta, -alpha);

            // STEP 5: Track the best move found
            if (score > bestScore)
            {
                bestScore = score;
                bestMove = move;
            }

            // STEP 6: Update alpha (our guaranteed minimum score)
            alpha = Math.Max(alpha, score);

            // STEP 7: PRUNING - If alpha >= beta, we can stop evaluating other moves
            // This is the core of Alpha-Beta pruning optimization
            // Why? Because the opponent would never let us reach this branch anyway
            // (they have a better option elsewhere), so continuing is pointless
            if (alpha >= beta)
                break;
        }

        return bestMove;
    }

    // ============================================
    // NEGAMAX ALGORITHM (RECURSIVE GAME TREE SEARCH)
    // ============================================

    /// <summary>
    /// NegaMax: The core recursive algorithm that evaluates the game tree.
    /// 
    /// WHAT IT DOES:
    /// - Recursively explores possible future moves
    /// - Returns a score representing how good this position is
    /// - Uses Alpha-Beta pruning to skip branches that don't matter
    /// 
    /// KEY INSIGHT (Why "Negamax"):
    /// Instead of having separate min/max functions like traditional minimax,
    /// NegaMax uses ONE function and NEGATES scores at each level:
    /// - Your best move from Player A's view = Your worst move from Player B's view
    /// - So we negate the recursive call: -NegaMax(...) 
    /// - This simplifies the algorithm while maintaining correctness
    /// 
    /// PARAMETERS:
    /// - board: Current board state to evaluate
    /// - depth: How many moves deep we still need to search (countdown to 0)
    /// - currentPlayer: Whose turn it is in this recursive call
    /// - alpha: Best score WE can guarantee (maximizing player's lower bound)
    /// - beta: Best score OPPONENT can guarantee (minimizing player's upper bound)
    /// 
    /// RETURNS:
    /// - Integer score: positive = good for currentPlayer, negative = bad for currentPlayer
    /// </summary>
    private int NegaMax(BoardState board, int depth, DiscColor currentPlayer, int alpha, int beta)
    {
        // ===== BASE CASE: Stop recursion when we've reached max depth =====
        // At depth 0, we can't look further ahead, so evaluate the position and return
        if (depth == 0)
        {
            return EvaluateBoard(board, currentPlayer);
        }

        // ===== MOVE GENERATION: Find all legal moves for this player =====
        var validMoves = GetValidMoves(board, currentPlayer);

        // ===== HANDLE PASS (No legal moves) =====
        if (validMoves.Count == 0)
        {
            // If current player can't move, check if opponent can move
            DiscColor opponent = currentPlayer == DiscColor.Black ? DiscColor.White : DiscColor.Black;
            var opponentMoves = GetValidMoves(board, opponent);

            // If opponent also can't move, the game is over - evaluate final position
            if (opponentMoves.Count == 0)
            {
                return EvaluateBoard(board, currentPlayer);
            }

            // If only current player can't move, pass the turn to opponent
            // Negate because we're switching to opponent's perspective
            return -NegaMax(board, depth - 1, opponent, -beta, -alpha);
        }

        // ===== RECURSIVE SEARCH: Evaluate all legal moves =====
        int maxScore = int.MinValue;  // Track the best score found

        foreach (var move in validMoves)
        {
            // Make a copy of the board and apply this move
            var newBoard = board.Clone();
            ApplyMove(newBoard, move, currentPlayer);

            // Switch perspective to opponent for recursive call
            DiscColor opponent = currentPlayer == DiscColor.Black ? DiscColor.White : DiscColor.Black;

            // Recursively search the resulting position
            // Negate because we switch from maximizing to minimizing perspective
            int score = -NegaMax(newBoard, depth - 1, opponent, -beta, -alpha);

            // Update the best score found so far
            maxScore = Math.Max(maxScore, score);
            alpha = Math.Max(alpha, score);

            // ===== ALPHA-BETA PRUNING =====
            // If alpha >= beta, this branch can be pruned:
            // - Alpha = best score we can guarantee
            // - Beta = best score opponent can guarantee
            // - If our best >= opponent's best, opponent won't let us reach this branch
            // - So we can safely skip evaluating remaining moves at this level
            if (alpha >= beta)
                break;  // PRUNING OCCURS HERE - major performance optimization
        }

        return maxScore;
    }

    // ============================================
    // BOARD EVALUATION FUNCTION
    // ============================================

    /// <summary>
    /// EvaluateBoard: Scores a board position without looking further ahead.
    /// This is called when we've reached our search depth limit.
    /// 
    /// STRATEGY: Positional Weighting
    /// Instead of just counting pieces, we weight squares by their strategic value.
    /// This is crucial for playing strong Othello - pure piece count is misleading
    /// because a position with fewer pieces in better squares is often winning.
    /// 
    /// POSITIONAL VALUE TABLE (Strategic importance):
    /// ```
    /// 400   -30   20   10   10   20  -30  400     <- Row 0: Corners and edges
    /// -30  -100   -5   -5   -5   -5 -100  -30     <- Row 1: AVOID - next to corners
    ///  20    -5    5    1    1    5   -5   20     <- Rows 2-5: Moderate zones
    ///  10    -5    1    1    1    1   -5   10
    ///  10    -5    1    1    1    1   -5   10
    ///  20    -5    5    1    1    5   -5   20
    /// -30  -100   -5   -5   -5   -5 -100  -30     <- Row 6: AVOID - next to corners
    /// 400   -30   20   10   10   20  -30  400     <- Row 7: Corners and edges
    /// ```
    /// 
    /// WHY THESE VALUES:
    /// - Corners (400): Game-winning strategic positions. Unflippable once secured.
    /// - Adjacent to corners (-100): Extremely risky! Giving opponent access to corner.
    /// - Edges (10-20): Good real estate, but flippable
    /// - Center (1-5): Neutral, constantly flipped during game
    /// 
    /// This weighting is based on professional Othello strategy literature
    /// and is widely used in competitive AI implementations.
    /// 
    /// PARAMETERS:
    /// - board: The board position to evaluate
    /// - color: We evaluate from this color's perspective (positive = good for them)
    /// 
    /// RETURNS:
    /// - Integer score: Sum of all piece positions' weights
    /// </summary>
    private int EvaluateBoard(BoardState board, DiscColor color)
    {
        int score = 0;
        DiscColor opponent = color == DiscColor.Black ? DiscColor.White : DiscColor.Black;

        // Define how valuable each square is (positional weighting table)
        // Higher number = more valuable
        // Negative number = don't want opponent to have it
        int[,] positionWeights = new int[8, 8]
        {
            // Corners worth 400 each - most valuable squares on board
            // Squares adjacent to corners worth -100 - heavy penalty (they help opponent reach corners)
            // Edges worth 10-20 - moderately valuable
            // Center worth 1-5 - neutral zone
            { 400,   -30,  20,  10,  10,  20,  -30,  400 },  // Top edge
            { -30,  -100,  -5,  -5,  -5,  -5, -100,  -30 },  // Danger zone next to corners
            {  20,   -5,   5,   1,   1,   5,   -5,   20 },   // Mid zones
            {  10,   -5,   1,   1,   1,   1,   -5,   10 },
            {  10,   -5,   1,   1,   1,   1,   -5,   10 },
            {  20,   -5,   5,   1,   1,   5,   -5,   20 },
            { -30,  -100,  -5,  -5,  -5,  -5, -100,  -30 },  // Danger zone next to corners
            { 400,   -30,  20,  10,  10,  20,  -30,  400 }   // Bottom edge
        };

        // Iterate through every square on the board
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                // If our color is on this square, ADD its weight to our score
                if (board.Grid[r, c] == color)
                {
                    score += positionWeights[r, c];
                }
                // If opponent's color is on this square, SUBTRACT its weight from our score
                // (same as adding the negative, but conceptually important)
                else if (board.Grid[r, c] == opponent)
                {
                    score -= positionWeights[r, c];
                }
                // If the square is empty (DiscColor.None), it contributes 0 to the score
            }
        }

        return score;
    }

    // ============================================
    // MOVE APPLICATION (BOARD MANIPULATION)
    // ============================================

    /// <summary>
    /// ApplyMove: Makes a move on a board and flips all affected opponent pieces.
    /// This is how we simulate moves during the game tree search.
    /// 
    /// OTHELLO RULES:
    /// 1. Place your piece on an empty square
    /// 2. Find all directions (8 possible: up, down, left, right, and 4 diagonals)
    /// 3. In each direction, if there's a continuous line of opponent pieces
    ///    followed by your own piece, flip all those opponent pieces to your color
    /// 
    /// IMPLEMENTATION:
    /// For each of 8 directions:
    /// - Count consecutive opponent pieces in that direction
    /// - If we find our own piece at the end of that line, flip all between
    /// - If no endpoint found, don't flip anything in that direction
    /// 
    /// PARAMETERS:
    /// - board: The board to modify (MODIFIED IN PLACE)
    /// - move: The position to place the new piece
    /// - color: The color of the piece being placed
    /// </summary>
    private void ApplyMove(BoardState board, Move move, DiscColor color)
    {
        // Safety check: verify the square is empty
        if (board.Grid[move.Row, move.Column] != DiscColor.None)
            return;  // Can't place piece on occupied square

        // Place the piece on the board
        board.Grid[move.Row, move.Column] = color;

        // Direction vectors for all 8 directions: up, down, left, right, 4 diagonals
        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };  // Row deltas
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };  // Column deltas

        DiscColor opponent = color == DiscColor.Black ? DiscColor.White : DiscColor.Black;

        // Check all 8 directions
        for (int i = 0; i < 8; i++)
        {
            // In this direction, collect all consecutive opponent pieces
            List<(int, int)> toFlip = new List<(int, int)>();
            int r = move.Row + dr[i];
            int c = move.Column + dc[i];

            // Walk in this direction, collecting opponent pieces
            while (r >= 0 && r < 8 && c >= 0 && c < 8 && board.Grid[r, c] == opponent)
            {
                toFlip.Add((r, c));  // Remember this piece might need flipping
                r += dr[i];
                c += dc[i];
            }

            // Check if this line ends with our own piece
            // If yes, we have a valid sandwich - flip all opponent pieces in between
            // If no, this direction contributes no flips
            if (r >= 0 && r < 8 && c >= 0 && c < 8 && board.Grid[r, c] == color && toFlip.Count > 0)
            {
                // Valid sandwich found! Flip all collected opponent pieces
                foreach (var (flipR, flipC) in toFlip)
                {
                    board.Grid[flipR, flipC] = color;
                }
            }
        }
    }

    // ============================================
    // MOVE GENERATION (FINDING LEGAL MOVES)
    // ============================================

    /// <summary>
    /// GetValidMoves: Finds all legal moves available for a player.
    /// Scans the entire board and checks each empty square.
    /// 
    /// PARAMETERS:
    /// - board: Current board state
    /// - color: Find moves for this color (Black or White)
    /// 
    /// RETURNS:
    /// - List of Move objects representing all legal moves
    /// 
    /// NOTE: This is called many times during the search, so performance matters.
    /// Even small optimizations here have big impact on total search speed.
    /// </summary>
    private List<Move> GetValidMoves(BoardState board, DiscColor color)
    {
        var moves = new List<Move>();

        // Check every square on the board
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                // If this square is empty and is a valid move, add it to the list
                if (IsValidMove(board, new Move(r, c), color))
                {
                    moves.Add(new Move(r, c));
                }
            }
        }
        return moves;
    }

    /// <summary>
    /// IsValidMove: Checks if a specific move is legal.
    /// A move is valid if it results in flipping at least one opponent piece.
    /// 
    /// ALGORITHM:
    /// 1. Check if target square is empty
    /// 2. Look in all 8 directions
    /// 3. In each direction, count consecutive opponent pieces
    /// 4. If that line is ended by our own piece, the move is valid
    /// 5. Return true if ANY direction results in flips
    /// 
    /// PARAMETERS:
    /// - board: Board to check
    /// - move: Position to validate
    /// - color: What color is attempting this move
    /// 
    /// RETURNS:
    /// - True if move is legal, False otherwise
    /// </summary>
    private bool IsValidMove(BoardState board, Move move, DiscColor color)
    {
        // Target square must be empty
        if (board.Grid[move.Row, move.Column] != DiscColor.None)
            return false;

        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };  // 8 directions
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };
        DiscColor opponent = color == DiscColor.Black ? DiscColor.White : DiscColor.Black;

        // Check all 8 directions
        for (int i = 0; i < 8; i++)
        {
            int r = move.Row + dr[i];
            int c = move.Column + dc[i];
            int count = 0;  // Count opponent pieces in this direction

            // Count consecutive opponent pieces
            while (r >= 0 && r < 8 && c >= 0 && c < 8 && board.Grid[r, c] == opponent)
            {
                r += dr[i];
                c += dc[i];
                count++;
            }

            // If we found opponent pieces AND a terminating piece of our color,
            // this direction results in valid flips
            if (r >= 0 && r < 8 && c >= 0 && c < 8 && board.Grid[r, c] == color && count > 0)
            {
                return true;  // At least one valid direction found
            }
        }

        return false;  // No valid direction found
    }
}