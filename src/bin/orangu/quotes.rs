// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

pub enum QuoteModule {
    None,
    StarTrek,
    StarWars,
    MarcoPierreWhite,
    GordonRamsay,
    CalvinAndHobbes,
    SunTzuMandarin,
    SunTzuEnglish,
    AttilaTheHun,
    All,
}

/// The quote-set names accepted in `[orangu].quotes`, in the order offered for
/// completion. Kept next to [`QuoteModule::from_str`], which parses them.
pub const QUOTE_OPTIONS: &[&str] = &[
    "none",
    "star_trek",
    "star_wars",
    "marco_pierre_white",
    "gordon_ramsay",
    "calvin_and_hobbes",
    "sun_tzu_mandarin",
    "sun_tzu_english",
    "attila_the_hun",
    "all",
];

impl QuoteModule {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "star_trek" => Self::StarTrek,
            "star_wars" => Self::StarWars,
            "marco_pierre_white" => Self::MarcoPierreWhite,
            "gordon_ramsay" => Self::GordonRamsay,
            "calvin_and_hobbes" => Self::CalvinAndHobbes,
            "sun_tzu_mandarin" => Self::SunTzuMandarin,
            "sun_tzu_english" => Self::SunTzuEnglish,
            "attila_the_hun" => Self::AttilaTheHun,
            "all" => Self::All,
            _ => Self::None,
        }
    }

    pub fn pick(&self, seed: u64) -> Option<&'static str> {
        let pool: &[&[&str]] = match self {
            Self::None => return None,
            Self::StarTrek => &[STAR_TREK],
            Self::StarWars => &[STAR_WARS],
            Self::MarcoPierreWhite => &[MARCO_PIERRE_WHITE],
            Self::GordonRamsay => &[GORDON_RAMSAY],
            Self::CalvinAndHobbes => &[CALVIN_AND_HOBBES],
            Self::SunTzuMandarin => &[SUN_TZU_MANDARIN],
            Self::SunTzuEnglish => &[SUN_TZU_ENGLISH],
            Self::AttilaTheHun => &[ATTILA_THE_HUN],
            Self::All => &[
                STAR_TREK,
                STAR_WARS,
                MARCO_PIERRE_WHITE,
                GORDON_RAMSAY,
                CALVIN_AND_HOBBES,
                SUN_TZU_MANDARIN,
                SUN_TZU_ENGLISH,
                ATTILA_THE_HUN,
            ],
        };
        let total: usize = pool.iter().map(|s| s.len()).sum();
        if total == 0 {
            return None;
        }
        let mut idx = seed as usize % total;
        for slice in pool {
            if idx < slice.len() {
                return Some(slice[idx]);
            }
            idx -= slice.len();
        }
        None
    }
}

static STAR_TREK: &[&str] = &[
    "Live long and prosper.",
    "Logic is the beginning of wisdom, not the end.",
    "The needs of the many outweigh the needs of the few.",
    "Space: the final frontier. These are the voyages of the starship Enterprise.",
    "Risk is our business. That's what this starship is all about.",
    "KHAAAANNN",
    "Make it so.",
    "Resistance is futile.",
    "He's dead, Jim.",
    "I'm a doctor, not a bricklayer.",
    "To boldly go where no man has gone before.",
    "Fascinating.",
    "Engage.",
    "It is possible to commit no mistakes and still lose. That is not a weakness. That is life.",
    "Things are only impossible until they're not.",
    "I canna' change the laws of physics.",
];

static STAR_WARS: &[&str] = &[
    "May the Force be with you.",
    "Do. Or do not. There is no try.",
    "I am your father.",
    "I've got a bad feeling about this.",
    "In my experience, there is no such thing as luck.",
    "The Force will be with you, always.",
    "That's no moon.",
    "Help me, Obi-Wan Kenobi. You're my only hope.",
    "It's a trap!",
    "Never tell me the odds.",
];

static MARCO_PIERRE_WHITE: &[&str] = &[
    "Cooking is the easy part. Doing it every day is the hard part.",
    "I cook for people, not for critics.",
    "The more you learn, the more you realize how little you know.",
    "Perfection is a lot of little things done well.",
    "Nature is the true artist; the chef is merely the cook.",
    "When you understand an ingredient, you understand the dish.",
    "Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for life.",
    "A tree without roots is just a piece of wood.",
    "If you are not extreme, then people will take shortcuts because they don't fear you.",
    "Strategy will compensate the talent. The talent will never compensate the strategy.",
    "Mother Nature is the true artist and our job as cooks is to allow her to shine.",
    "Success is born out of arrogance, but greatness comes from humility.",
    "Once you accept you are being judged by people who have less knowledge than yourself, then what's it worth?",
    "People who can give themselves every day. They're the people that I admire, they're real people.",
    "A person who works with their hands is a labourer, a person who works with their hands and their brain is a craftsman, a person who works with their hands, their brain and their heart is an artist. Ask yourself, who are you?",
    "Taste. Taste. Taste.",
    "Keep it simple.",
    "Look at each individual ingredient for what it is, not for what you want it to be.",
    "If you look at my hand what do you see ? A palm. I see four knuckles. We are looking at the same thing, but we are seeing something very different.",
    "Never allow the obvious to blind you.",
];

static GORDON_RAMSAY: &[&str] = &[
    "This chicken is so raw it's still asking why it crossed the road.",
    "A recipe has no soul. You, as the cook, must bring soul to the recipe.",
    "Cooking today is a young man's game — I don't give a bollocks what anyone says.",
    "I don't like looking back. I'm always constantly looking forward.",
    "Kitchens are hard environments and they form incredibly strong characters.",
    "I taught myself to cook — that is what makes me a truly self-made chef.",
    "The secret of a good restaurant: find great ingredients and don't ruin them.",
];

static CALVIN_AND_HOBBES: &[&str] = &[
    "It's a magical world, Hobbes, ol' buddy... let's go exploring!",
    "Sometimes I think the surest sign that intelligent life exists elsewhere in the universe is that none of it has tried to contact us.",
    "Reality continues to ruin my life.",
    "I go to school, but I never learn what I want to know.",
    "Life is like topography, Hobbes. There are summits of happiness and success, flat stretches of boring routine, and valleys of frustration and failure.",
    "I'm not dumb, I just have a command of thoroughly useless information.",
    "You can't just turn on creativity like a faucet. You have to be in the right mood. What mood is that? Last-minute panic.",
    "Weekends don't count unless you spend them doing something completely pointless.",
];

// Sun Tzu, The Art of War (孫子兵法).
static SUN_TZU_MANDARIN: &[&str] = &[
    "兵者，國之大事，死生之地，存亡之道，不可不察也。",
    "知彼知己，百戰不殆。",
    "兵者，詭道也。",
    "不戰而屈人之兵，善之善者也。",
    "上兵伐謀，其次伐交，其次伐兵，其下攻城。",
    "故善戰者，致人而不致於人。",
    "兵貴勝，不貴久。",
    "攻其無備，出其不意。",
    "善戰者，求之於勢，不責於人。",
    "故善用兵者，屈人之兵而非戰也。",
    "勝兵先勝而後求戰，敗兵先戰而後求勝。",
    "故善戰者，立於不敗之地，而不失敵之敗也。",
    "善守者，藏於九地之下；善攻者，動於九天之上。",
    "凡戰者，以正合，以奇勝。",
    "故知勝有五：知可以戰與不可以戰者勝。",
    "其疾如風，其徐如林，侵掠如火，不動如山。",
];

static SUN_TZU_ENGLISH: &[&str] = &[
    "The art of war is of vital importance to the State.",
    "If you know the enemy and know yourself, you need not fear the result of a hundred battles.",
    "All warfare is based on deception.",
    "To subdue the enemy without fighting is the acme of skill.",
    "The supreme art of war is to subdue the enemy without fighting.",
    "The skilful fighter imposes his will on the enemy, but does not allow the enemy's will to be imposed on him.",
    "In war, then, let your great object be victory, not lengthy campaigns.",
    "Attack him where he is unprepared, appear where you are not expected.",
    "The clever combatant looks to the effect of combined energy, and does not require too much from individuals.",
    "The skilful leader subdues the enemy's troops without any fighting.",
    "Victorious warriors win first and then go to war, while defeated warriors go to war first and then seek to win.",
    "The good fighter is able to secure himself against defeat, but cannot make certain of defeating the enemy.",
    "He who is skilled in defence hides in the most secret recesses of the earth; he who is skilled in attack flashes forth from the topmost heights of heaven.",
    "In all fighting, the direct method may be used for joining battle, but indirect methods will be needed in order to secure victory.",
    "He will win who knows when to fight and when not to fight.",
    "Let your rapidity be that of the wind, your compactness that of the forest. In raiding and plundering be like fire, in immovability like a mountain.",
];

// Attila the Hun, in the leadership tradition of Wess Roberts' "Leadership
// Secrets of Attila the Hun".
static ATTILA_THE_HUN: &[&str] = &[
    "There, where I have passed, the grass will never grow again.",
    "Chieftains seldom err in being too kind; however, they often err in being too lenient.",
    "A king without a people is no king at all.",
    "Leaders must teach their Huns well that which is expected of them. Otherwise, Huns will probably do something not expected of them.",
    "The greatness of a Hun is measured by the sacrifices he is willing to make for the good of the nation.",
    "It is unfortunate when final decisions are made by chieftains headquartered miles away from the front, where they can only guess at conditions and potentialities known only to the captain of the battle.",
    "Wise chieftains never take themselves too seriously.",
    "Superficial leadership will not work in the heat of battle, nor in the heat of life.",
    "Huns should never be made to feel that they are alone in their struggle, even when on a lonely and arduous mission.",
    "A wise chieftain never depends on luck. Rather, he always tips chance in his favor.",
    "Adequate compensation for a year's loyalty is a year's pay. Loyalty cannot be bought; it must be earned.",
    "Strong, able, willing Huns with marginal skills can be taught to perfect their abilities. The greater task is to teach them humility and to control their fanaticism.",
    "It is not enough that I succeed, everybody else must fail.",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_returns_no_quote() {
        assert!(QuoteModule::None.pick(0).is_none());
        assert!(QuoteModule::None.pick(42).is_none());
    }

    #[test]
    fn each_module_returns_quotes() {
        assert!(QuoteModule::StarTrek.pick(0).is_some());
        assert!(QuoteModule::StarWars.pick(1).is_some());
        assert!(QuoteModule::MarcoPierreWhite.pick(2).is_some());
        assert!(QuoteModule::GordonRamsay.pick(3).is_some());
        assert!(QuoteModule::CalvinAndHobbes.pick(4).is_some());
        assert!(QuoteModule::SunTzuMandarin.pick(5).is_some());
        assert!(QuoteModule::SunTzuEnglish.pick(6).is_some());
        assert!(QuoteModule::AttilaTheHun.pick(7).is_some());
    }

    #[test]
    fn all_covers_every_slot() {
        let total: usize = [
            STAR_TREK,
            STAR_WARS,
            MARCO_PIERRE_WHITE,
            GORDON_RAMSAY,
            CALVIN_AND_HOBBES,
            SUN_TZU_MANDARIN,
            SUN_TZU_ENGLISH,
            ATTILA_THE_HUN,
        ]
        .iter()
        .map(|s| s.len())
        .sum();
        for seed in 0..total as u64 {
            assert!(QuoteModule::All.pick(seed).is_some());
        }
    }

    #[test]
    fn from_str_parses_all_variants() {
        assert!(matches!(QuoteModule::from_str("none"), QuoteModule::None));
        assert!(matches!(
            QuoteModule::from_str("star_trek"),
            QuoteModule::StarTrek
        ));
        assert!(matches!(
            QuoteModule::from_str("star_wars"),
            QuoteModule::StarWars
        ));
        assert!(matches!(
            QuoteModule::from_str("marco_pierre_white"),
            QuoteModule::MarcoPierreWhite
        ));
        assert!(matches!(
            QuoteModule::from_str("gordon_ramsay"),
            QuoteModule::GordonRamsay
        ));
        assert!(matches!(
            QuoteModule::from_str("calvin_and_hobbes"),
            QuoteModule::CalvinAndHobbes
        ));
        assert!(matches!(
            QuoteModule::from_str("sun_tzu_mandarin"),
            QuoteModule::SunTzuMandarin
        ));
        assert!(matches!(
            QuoteModule::from_str("sun_tzu_english"),
            QuoteModule::SunTzuEnglish
        ));
        assert!(matches!(
            QuoteModule::from_str("attila_the_hun"),
            QuoteModule::AttilaTheHun
        ));
        assert!(matches!(QuoteModule::from_str("all"), QuoteModule::All));
        assert!(matches!(
            QuoteModule::from_str("unknown"),
            QuoteModule::None
        ));
    }
}
